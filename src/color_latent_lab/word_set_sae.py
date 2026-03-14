from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from .analysis_common import cosine_similarity_matrix
from .model_utils import _resolve_device
from .run_support import HeartbeatRecorder, _read_json, _write_json, _write_jsonl
from .sae_geometry import (
    CORE_COLOR_FAMILIES,
    QWEN_OFF_THE_SHELF_SAE_LAYERS,
    QWEN_OFF_THE_SHELF_SAE_REPO,
    _capture_last_token_activations,
    _encode_activations,
    _require_geometry_stack,
    load_off_the_shelf_sae,
)

COMMON_COLOR_FAMILY_WORDS: tuple[str, ...] = CORE_COLOR_FAMILIES


def _sanitize_label_for_filename(value: str) -> str:
    cleaned = [
        character.lower() if character.isalnum() else "_"
        for character in value.strip()
    ]
    collapsed = "".join(cleaned).strip("_")
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")
    return collapsed or "item"


def _build_prompt_rows(words: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, word in enumerate(words):
        rows.append(
            {
                "prompt": word,
                "record_id": f"word-{index:04d}-{_sanitize_label_for_filename(word)}",
                "schema": "word",
                "value": word,
                "word": word,
            }
        )
    return rows


def _leave_one_out_feature_vectors(np: Any, values: Any) -> tuple[Any, Any]:
    if int(values.shape[0]) < 2:
        raise ValueError("Need at least two feature vectors to build leave-one-out contrasts.")
    values_32 = values.astype(np.float32)
    total = values_32.sum(axis=0, keepdims=True).astype(np.float32)
    mean_others = ((total - values_32) / float(values_32.shape[0] - 1)).astype(np.float32)
    return mean_others, (values_32 - mean_others).astype(np.float32)


def _write_layer_outputs(
    *,
    np: Any,
    layer_dir: Path,
    words: tuple[str, ...],
    rows: list[dict[str, Any]],
    residual_vectors: Any,
    feature_vectors: Any,
) -> dict[str, Any]:
    layer_dir.mkdir(parents=True, exist_ok=True)
    np.save(layer_dir / "residual_vectors.npy", residual_vectors.astype(np.float32))
    np.save(layer_dir / "feature_vectors.npy", feature_vectors.astype(np.float32))

    mean_feature_vector = feature_vectors.mean(axis=0).astype(np.float32)
    mean_other_vectors, leave_one_out_vectors = _leave_one_out_feature_vectors(np, feature_vectors)
    similarity = cosine_similarity_matrix(leave_one_out_vectors).astype(np.float32)

    np.save(layer_dir / "mean_feature_vector.npy", mean_feature_vector)
    np.save(layer_dir / "mean_other_feature_vectors.npy", mean_other_vectors.astype(np.float32))
    np.save(layer_dir / "leave_one_out_feature_vectors.npy", leave_one_out_vectors.astype(np.float32))
    np.save(layer_dir / "cosine_similarity_matrix.npy", similarity)

    _write_json(
        layer_dir / "cosine_similarity_matrix.json",
        {
            "layer": int(layer_dir.name.split("_", 1)[1]),
            "matrix": similarity.tolist(),
            "words": list(words),
        },
    )

    word_rows: list[dict[str, Any]] = []
    for row_index, word in enumerate(words):
        word_rows.append(
            {
                "feature_vector_norm": float(np.linalg.norm(feature_vectors[row_index])),
                "last_token_id": rows[row_index].get("last_token_id"),
                "last_token_text": rows[row_index].get("last_token_text"),
                "leave_one_out_norm": float(np.linalg.norm(leave_one_out_vectors[row_index])),
                "mean_other_norm": float(np.linalg.norm(mean_other_vectors[row_index])),
                "record_id": rows[row_index]["record_id"],
                "word": word,
                "word_rank": row_index,
            }
        )
    _write_jsonl(layer_dir / "word_vectors.jsonl", word_rows)

    summary = {
        "feature_count": int(feature_vectors.shape[1]),
        "layer": int(layer_dir.name.split("_", 1)[1]),
        "mean_abs_offdiag_cosine": float(
            np.mean(np.abs(similarity[np.triu_indices(len(words), k=1)]))
        ),
        "mean_feature_vector_norm": float(np.linalg.norm(mean_feature_vector)),
        "mean_leave_one_out_norm": float(np.linalg.norm(leave_one_out_vectors, axis=1).mean()),
        "word_count": len(words),
    }
    _write_json(layer_dir / "summary.json", summary)
    return summary


def _write_report(path: Path, *, model_name: str, layers: tuple[int, ...], words: tuple[str, ...]) -> None:
    lines = [
        "# SAE color-family feature-space matrices",
        "",
        f"- Model: `{model_name}`",
        f"- Words: `{list(words)}`",
        f"- Layers: `{list(layers)}`",
        "",
        "Per-layer artifacts:",
        "",
        "- `manifest.json`",
        "- `heartbeat_status.json`",
        "- `heartbeat_events.jsonl`",
        "- `panel.jsonl`",
        "- `heatmaps/index.html`",
        "- `heatmaps/layer_XX_cosine_heatmap.svg`",
        "- `layer_XX/residual_vectors.npy`",
        "- `layer_XX/feature_vectors.npy`",
        "- `layer_XX/mean_feature_vector.npy`",
        "- `layer_XX/mean_other_feature_vectors.npy`",
        "- `layer_XX/leave_one_out_feature_vectors.npy`",
        "- `layer_XX/cosine_similarity_matrix.json` / `.npy`",
        "- `layer_XX/word_vectors.jsonl`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_labeled_similarity_matrix(*, layer: int, words: tuple[str, ...], matrix: Any) -> str:
    row_label_width = max(len(word) for word in words)
    cell_width = max(8, max(len(word) for word in words))
    header = " " * (row_label_width + 2) + " ".join(f"{word:>{cell_width}}" for word in words)
    lines = [f"Layer {layer} cosine similarity matrix", header]
    for row_index, word in enumerate(words):
        values = " ".join(f"{float(matrix[row_index, col_index]):>{cell_width}.3f}" for col_index in range(len(words)))
        lines.append(f"{word:>{row_label_width}}  {values}")
    return "\n".join(lines)


def _write_final_results(output_dir: Path, *, summary: dict[str, Any]) -> None:
    payload = {
        "kind": "sae_color_family_feature_matrix",
        "key_artifacts": {
            "heatmap_index": "heatmaps/index.html",
            "heartbeat_events": "heartbeat_events.jsonl",
            "heartbeat_status": "heartbeat_status.json",
            "layer_summary": "layer_summary.jsonl",
            "manifest": "manifest.json",
            "panel": "panel.jsonl",
            "report": "report.md",
            "summary": "summary.json",
        },
        "summary": summary,
    }
    _write_json(output_dir / "final_results.json", payload)


def _blend_channel(start: int, end: int, fraction: float) -> int:
    bounded = max(0.0, min(1.0, float(fraction)))
    return int(round(start + (end - start) * bounded))


def _hex_color(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{max(0, min(255, channel)):02x}" for channel in rgb)


def _blend_colors(start: tuple[int, int, int], end: tuple[int, int, int], fraction: float) -> str:
    return _hex_color(
        tuple(_blend_channel(start[channel], end[channel], fraction) for channel in range(3))
    )


def _heatmap_fill(value: float) -> str:
    neutral = (245, 241, 234)
    positive = (198, 92, 44)
    negative = (48, 103, 166)
    bounded = max(-1.0, min(1.0, float(value)))
    magnitude = abs(bounded) ** 0.9
    if bounded >= 0.0:
        return _blend_colors(neutral, positive, magnitude)
    return _blend_colors(neutral, negative, magnitude)


def _heatmap_text_fill(value: float) -> str:
    return "#fffaf2" if abs(float(value)) >= 0.58 else "#231c16"


def _write_similarity_heatmap_svg(
    path: Path,
    *,
    layer: int,
    words: tuple[str, ...],
    matrix: Any,
) -> None:
    cell_size = 54.0
    label_padding = 26.0
    max_label_len = max(len(word) for word in words)
    left_margin = max(150.0, max_label_len * 10.0 + 54.0)
    top_margin = max(188.0, max_label_len * 9.0 + 88.0)
    right_margin = 54.0
    bottom_margin = 112.0
    grid_size = len(words) * cell_size
    width = left_margin + grid_size + right_margin
    height = top_margin + grid_size + bottom_margin
    legend_width = min(320.0, grid_size)
    legend_x = left_margin
    legend_y = top_margin + grid_size + 52.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" viewBox="0 0 {int(width)} {int(height)}">',
        "<defs>",
        '<linearGradient id="heatmap-scale" x1="0%" y1="0%" x2="100%" y2="0%">',
        '<stop offset="0%" stop-color="#3067a6" />',
        '<stop offset="50%" stop-color="#f5f1ea" />',
        '<stop offset="100%" stop-color="#c65c2c" />',
        "</linearGradient>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#fbfaf7" />',
        (
            f'<text x="{left_margin}" y="36" font-family="Helvetica, Arial, sans-serif" font-size="26" '
            f'fill="#111111">Layer {layer} cosine similarity heatmap</text>'
        ),
        (
            f'<text x="{left_margin}" y="62" font-family="Helvetica, Arial, sans-serif" font-size="14" '
            f'fill="#555555">Leave-one-out SAE feature vectors for the common color-family words</text>'
        ),
        (
            f'<text x="{left_margin}" y="84" font-family="Helvetica, Arial, sans-serif" font-size="12" '
            f'fill="#6b6257">Cells are cosine similarities in SAE feature space, scaled from -1 to 1.</text>'
        ),
        (
            f'<rect x="{left_margin}" y="{top_margin}" width="{grid_size}" height="{grid_size}" '
            f'fill="none" stroke="#c7beb1" stroke-width="1.2" />'
        ),
    ]
    for index, word in enumerate(words):
        x = left_margin + index * cell_size + cell_size / 2.0
        y = top_margin + index * cell_size + cell_size / 2.0
        lines.append(
            f'<text x="{left_margin - label_padding}" y="{y + 5.0:.2f}" text-anchor="end" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#2d261f">{html.escape(word)}</text>'
        )
        lines.append(
            f'<text x="{x:.2f}" y="{top_margin - 14.0}" transform="rotate(-42 {x:.2f} {top_margin - 14.0:.2f})" '
            f'text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#2d261f">{html.escape(word)}</text>'
        )
    for row_index, row_word in enumerate(words):
        for column_index, column_word in enumerate(words):
            value = float(matrix[row_index, column_index])
            x = left_margin + column_index * cell_size
            y = top_margin + row_index * cell_size
            fill = _heatmap_fill(value)
            stroke = "#8f8374" if row_index == column_index else "#ddd4c8"
            stroke_width = "1.2" if row_index == column_index else "0.8"
            lines.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_size:.2f}" height="{cell_size:.2f}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}">'
                f"<title>{html.escape(f'Layer {layer} | {row_word} vs {column_word} | cosine={value:.3f}')}</title></rect>"
            )
            lines.append(
                f'<text x="{x + cell_size / 2.0:.2f}" y="{y + cell_size / 2.0 + 4.5:.2f}" text-anchor="middle" '
                f'font-family="Helvetica, Arial, sans-serif" font-size="11" fill="{_heatmap_text_fill(value)}">{value:.2f}</text>'
            )
    lines.extend(
        [
            f'<rect x="{legend_x}" y="{legend_y}" width="{legend_width}" height="16" fill="url(#heatmap-scale)" stroke="#c7beb1" stroke-width="1" />',
            f'<text x="{legend_x}" y="{legend_y + 34}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#555555">-1.0</text>',
            f'<text x="{legend_x + legend_width / 2.0:.2f}" y="{legend_y + 34}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#555555">0.0</text>',
            f'<text x="{legend_x + legend_width:.2f}" y="{legend_y + 34}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#555555">1.0</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_heatmap_index_html(
    path: Path,
    *,
    words: tuple[str, ...],
    layer_files: list[tuple[int, str]],
) -> None:
    word_line = ", ".join(words)
    cards = "\n".join(
        (
            '<article class="card">'
            f"<h2>Layer {layer}</h2>"
            f'<img src="{html.escape(filename)}" alt="Layer {layer} cosine similarity heatmap" />'
            "</article>"
        )
        for layer, filename in layer_files
    )
    document = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>SAE Word-Set Heatmaps</title>
    <style>
      :root {{
        color-scheme: light;
        --paper: #faf8f3;
        --panel: #fffdf8;
        --ink: #1f1a15;
        --muted: #62584c;
        --border: #d9d0c3;
      }}
      body {{
        margin: 0;
        padding: 28px;
        background: linear-gradient(180deg, #f8f5ed 0%, #fcfbf8 100%);
        color: var(--ink);
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      }}
      main {{
        max-width: 1500px;
        margin: 0 auto;
      }}
      h1 {{
        margin: 0 0 8px 0;
        font-size: 32px;
      }}
      p {{
        margin: 0 0 12px 0;
        color: var(--muted);
        font-size: 15px;
        line-height: 1.5;
      }}
      .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(560px, 1fr));
        gap: 20px;
        margin-top: 24px;
      }}
      .card {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 14px 32px rgba(84, 70, 46, 0.08);
      }}
      .card h2 {{
        margin: 0 0 12px 0;
        font-size: 20px;
      }}
      .card img {{
        display: block;
        width: 100%;
        height: auto;
        border-radius: 10px;
        border: 1px solid #e6ddd1;
        background: #fbfaf7;
      }}
      code {{
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 0.92em;
      }}
    </style>
  </head>
  <body>
    <main>
      <h1>SAE Word-Set Cosine Heatmaps</h1>
      <p>Each panel shows the cosine similarity matrix over leave-one-out SAE feature vectors for one layer.</p>
      <p><strong>Words:</strong> <code>{html.escape(word_line)}</code></p>
      <div class="grid">
        {cards}
      </div>
    </main>
  </body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def render_word_set_sae_heatmaps(
    *,
    run_dir: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    np, _torch, _ = _require_geometry_stack()
    resolved_run_dir = run_dir.resolve()
    resolved_output_dir = (resolved_run_dir / "heatmaps") if output_dir is None else output_dir.resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = resolved_run_dir / "summary.json"
    summary_payload = _read_json(summary_path) if summary_path.exists() else {}
    layers = tuple(
        sorted(
            int(path.name.split("_", 1)[1])
            for path in resolved_run_dir.glob("layer_*")
            if (path / "cosine_similarity_matrix.json").exists()
        )
    )
    if not layers:
        raise FileNotFoundError(f"No layer_XX/cosine_similarity_matrix.json files found under {resolved_run_dir}")
    layer_files: list[tuple[int, str]] = []
    canonical_words: tuple[str, ...] | None = None
    for layer in layers:
        payload = _read_json(resolved_run_dir / f"layer_{layer:02d}" / "cosine_similarity_matrix.json")
        words = tuple(str(word) for word in payload["words"])
        matrix = np.array(payload["matrix"], dtype=np.float32)
        if matrix.shape != (len(words), len(words)):
            raise ValueError(
                f"Layer {layer} matrix shape {matrix.shape} does not match word count {len(words)}."
            )
        if canonical_words is None:
            canonical_words = words
        elif words != canonical_words:
            raise ValueError(f"Layer {layer} words do not match the earlier cosine matrix word order.")
        filename = f"layer_{layer:02d}_cosine_heatmap.svg"
        _write_similarity_heatmap_svg(
            resolved_output_dir / filename,
            layer=layer,
            words=words,
            matrix=matrix,
        )
        layer_files.append((layer, filename))
    assert canonical_words is not None
    _write_heatmap_index_html(
        resolved_output_dir / "index.html",
        words=canonical_words,
        layer_files=layer_files,
    )
    render_summary = {
        "heatmap_count": len(layer_files),
        "layers": list(layers),
        "output_dir": str(resolved_output_dir),
        "run_dir": str(resolved_run_dir),
        "words": list(canonical_words),
    }
    if summary_payload:
        render_summary["run_summary"] = summary_payload
    _write_json(resolved_output_dir / "summary.json", render_summary)
    return render_summary


def run_word_set_sae_feature_experiment(
    *,
    output_dir: Path,
    model_name: str,
    sae_repo_id_or_path: str = QWEN_OFF_THE_SHELF_SAE_REPO,
    sae_layers: tuple[int, ...] = QWEN_OFF_THE_SHELF_SAE_LAYERS,
    trainer_index: int = 0,
    cache_dir: Path | None = None,
    batch_size: int = 64,
    encode_batch_size: int = 256,
    max_length: int = 16,
    device: str = "auto",
) -> dict[str, Any]:
    np, torch, _ = _require_geometry_stack()
    output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = HeartbeatRecorder(output_dir, label="sae-word-sets")
    phase = "setup"
    try:
        words = tuple(str(word) for word in COMMON_COLOR_FAMILY_WORDS)
        layers = tuple(int(layer) for layer in sae_layers)
        if len(words) < 2:
            raise ValueError("Need at least two common color-family words to build leave-one-out contrasts.")
        prompt_rows = _build_prompt_rows(words)
        heartbeat.write_manifest(
            command="sae-word-sets",
            model_name=model_name,
            sae_layers=list(layers),
            sae_repo_id_or_path=sae_repo_id_or_path,
            word_count=len(words),
            words=list(words),
        )
        heartbeat.update(
            phase=phase,
            message="Preparing SAE word-set color matrix run",
            layer_count=len(layers),
            word_count=len(words),
        )
        runtime_device = _resolve_device(torch, device)

        phase = "collect"
        heartbeat.update(
            phase=phase,
            message="Collecting residual activations for common color-family words",
            layer_count=len(layers),
            total_records=len(prompt_rows),
        )
        activations_by_layer, enriched_rows = _capture_last_token_activations(
            model_name=model_name,
            records=prompt_rows,
            layers=layers,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
            heartbeat=heartbeat,
        )
        _write_jsonl(output_dir / "panel.jsonl", enriched_rows)
        heartbeat.update(
            phase=phase,
            message="Collected residual activations for common color-family words",
            layer_count=len(layers),
            processed_records=len(enriched_rows),
            total_records=len(enriched_rows),
        )

        layer_summaries: list[dict[str, Any]] = []
        for layer in layers:
            phase = "analyze"
            heartbeat.update(
                phase=phase,
                message=f"Loading SAE for layer {layer}",
                current_layer=layer,
            )
            sae, _sae_config = load_off_the_shelf_sae(
                layer=layer,
                repo_id_or_path=sae_repo_id_or_path,
                trainer_index=trainer_index,
                device=str(runtime_device),
                cache_dir=cache_dir,
            )
            phase = "encode"
            heartbeat.update(
                phase=phase,
                message=f"Encoding SAE feature vectors at layer {layer}",
                current_layer=layer,
                activation_count=int(activations_by_layer[layer].shape[0]),
            )
            feature_vectors = _encode_activations(
                torch=torch,
                sae=sae,
                activations=activations_by_layer[layer],
                batch_size=encode_batch_size,
                device=str(runtime_device),
            )
            phase = "summarize"
            layer_summary = _write_layer_outputs(
                np=np,
                layer_dir=output_dir / f"layer_{layer:02d}",
                words=words,
                rows=enriched_rows,
                residual_vectors=activations_by_layer[layer],
                feature_vectors=feature_vectors,
            )
            layer_summaries.append(layer_summary)
            heartbeat.update(
                phase=phase,
                message=f"Wrote cosine similarity outputs for layer {layer}",
                current_layer=layer,
                feature_count=int(feature_vectors.shape[1]),
                mean_abs_offdiag_cosine=float(layer_summary["mean_abs_offdiag_cosine"]),
            )

        layer_summaries.sort(key=lambda row: int(row["layer"]))
        _write_jsonl(output_dir / "layer_summary.jsonl", layer_summaries)
        summary = {
            "layers": list(layers),
            "model_name": model_name,
            "sae_repo_id_or_path": sae_repo_id_or_path,
            "trainer_index": trainer_index,
            "word_count": len(words),
            "words": list(words),
        }
        _write_json(output_dir / "summary.json", summary)
        phase = "render"
        heartbeat.update(
            phase=phase,
            message="Rendering cosine heatmaps",
            layer_count=len(layers),
        )
        heatmap_summary = render_word_set_sae_heatmaps(run_dir=output_dir)
        heartbeat.update(
            phase=phase,
            message="Rendered cosine heatmaps",
            heatmap_count=int(heatmap_summary["heatmap_count"]),
        )
        _write_report(output_dir / "report.md", model_name=model_name, layers=layers, words=words)
        _write_final_results(output_dir, summary=summary)
        matrix_sections = []
        for layer in layers:
            matrix = np.load(output_dir / f"layer_{layer:02d}" / "cosine_similarity_matrix.npy").astype(np.float32)
            matrix_sections.append(_format_labeled_similarity_matrix(layer=layer, words=words, matrix=matrix))
        print("\n\n".join(matrix_sections), flush=True)
        heartbeat.update(phase="complete", message="SAE color-family matrix run complete", state="completed")
        return summary
    except Exception as error:
        heartbeat.fail(phase=phase, error=error)
        raise


__all__ = [
    "COMMON_COLOR_FAMILY_WORDS",
    "render_word_set_sae_heatmaps",
    "run_word_set_sae_feature_experiment",
]
