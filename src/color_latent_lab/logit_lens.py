from __future__ import annotations

import html
import math
from collections import Counter
from pathlib import Path
from typing import Any

from . import experiment as exp

FAMILY_EXEMPLAR_HEX: dict[str, str] = {
    "red": "#ff0000",
    "orange": "#ff8800",
    "yellow": "#ffff00",
    "green": "#00ff00",
    "cyan": "#00ffff",
    "blue": "#0000ff",
    "purple": "#8000ff",
    "magenta": "#ff00ff",
    "brown": "#8b4513",
    "black": "#000000",
    "white": "#ffffff",
    "gray": "#808080",
}
FAMILY_EXEMPLAR_RGB: dict[str, str] = {
    "red": "255,0,0",
    "orange": "255,136,0",
    "yellow": "255,255,0",
    "green": "0,255,0",
    "cyan": "0,255,255",
    "blue": "0,0,255",
    "purple": "128,0,255",
    "magenta": "255,0,255",
    "brown": "139,69,19",
    "black": "0,0,0",
    "white": "255,255,255",
    "gray": "128,128,128",
}
GROUP_COLORS: dict[str, str] = {
    "matched_family_mass": "#1f6f5f",
    "word_mass": "#9b3d7a",
    "hex_mass": "#b86910",
    "rgb_mass": "#2e6fd8",
    "family_accuracy": "#444444",
}
SEMANTIC_ONSET_FRACTION = 0.75
SEMANTIC_ONSET_MIN_ACCURACY = 0.2


def _logit_lens_checkpoint_dir(output_dir: Path, format_name: str) -> Path:
    return output_dir / "checkpoints" / "logit_lens_batches" / format_name


def _logit_lens_batch_rows_path(output_dir: Path, format_name: str, batch_index: int) -> Path:
    return _logit_lens_checkpoint_dir(output_dir, format_name) / f"batch_{batch_index:04d}.rows.jsonl"


def _logit_lens_batch_predictions_path(output_dir: Path, format_name: str, batch_index: int) -> Path:
    return (
        _logit_lens_checkpoint_dir(output_dir, format_name)
        / f"batch_{batch_index:04d}.predictions.jsonl"
    )


def _save_logit_lens_batch_checkpoint(
    *,
    output_dir: Path,
    format_name: str,
    batch_index: int,
    prediction_rows: list[dict[str, Any]],
    lens_rows: list[dict[str, Any]],
) -> None:
    checkpoint_dir = _logit_lens_checkpoint_dir(output_dir, format_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    exp._write_jsonl(
        _logit_lens_batch_predictions_path(output_dir, format_name, batch_index),
        prediction_rows,
    )
    exp._write_jsonl(_logit_lens_batch_rows_path(output_dir, format_name, batch_index), lens_rows)


def _load_logit_lens_batch_checkpoint(
    *,
    output_dir: Path,
    format_name: str,
    batch_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    predictions_path = _logit_lens_batch_predictions_path(output_dir, format_name, batch_index)
    rows_path = _logit_lens_batch_rows_path(output_dir, format_name, batch_index)
    if not predictions_path.exists() or not rows_path.exists():
        return None
    return exp._read_prediction_rows(predictions_path), exp._read_prediction_rows(rows_path)


def _get_output_embedding(model: Any) -> Any:
    getter = getattr(model, "get_output_embeddings", None)
    if callable(getter):
        embedding = getter()
        if embedding is not None:
            return embedding
    embedding = getattr(model, "lm_head", None)
    if embedding is not None:
        return embedding
    raise RuntimeError("Could not find output embedding / lm_head for logit lens.")


def _find_final_norm(model: Any) -> Any | None:
    for candidate in (
        getattr(getattr(model, "model", None), "norm", None),
        getattr(getattr(model, "model", None), "final_layernorm", None),
        getattr(getattr(model, "transformer", None), "ln_f", None),
        getattr(getattr(model, "gpt_neox", None), "final_layer_norm", None),
    ):
        if candidate is not None:
            return candidate
    return None


def _apply_final_norm(hidden: Any, final_norm: Any | None) -> Any:
    if final_norm is None:
        return hidden
    return final_norm(hidden)


def _encode_first_token_id(tokenizer: Any, literal: str) -> int | None:
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        token_ids = encode(literal, add_special_tokens=False)
        if token_ids:
            return int(token_ids[0])
    return None


def _decode_token(tokenizer: Any, token_id: int) -> str:
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        try:
            return str(
                decode([token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            ).replace("\n", "\\n")
        except TypeError:
            return str(decode([token_id])).replace("\n", "\\n")
    return str(token_id)


def _family_word_literals(family: str) -> list[str]:
    literals = [family, f" {family}", family.title(), f" {family.title()}"]
    if family == "gray":
        literals.extend(["grey", " grey", "Grey", " Grey"])
    return literals


def _build_token_groups(tokenizer: Any) -> dict[str, Any]:
    family_ids: dict[str, list[int]] = {}
    for family in FAMILY_EXEMPLAR_HEX:
        ids = {
            token_id
            for literal in _family_word_literals(family)
            if (token_id := _encode_first_token_id(tokenizer, literal)) is not None
        }
        family_ids[family] = sorted(ids)
    format_ids = {
        "word": sorted({token_id for ids in family_ids.values() for token_id in ids}),
        "hex": sorted(
            {
                token_id
                for exemplar in FAMILY_EXEMPLAR_HEX.values()
                for literal in (exemplar, f" {exemplar}")
                if (token_id := _encode_first_token_id(tokenizer, literal)) is not None
            }
        ),
        "rgb": sorted(
            {
                token_id
                for exemplar in FAMILY_EXEMPLAR_RGB.values()
                for literal in (exemplar, f" {exemplar}")
                if (token_id := _encode_first_token_id(tokenizer, literal)) is not None
            }
        ),
    }
    return {"family_ids": family_ids, "format_ids": format_ids}


def _group_probability_mass(torch: Any, logits: Any, log_norm: Any, token_ids: list[int]) -> Any | None:
    if not token_ids:
        return None
    subset = logits[:, token_ids]
    return torch.exp(torch.logsumexp(subset, dim=-1) - log_norm)


def _top_tokens_per_row(*, tokenizer: Any, top_ids: Any, top_values: Any) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    for token_ids, values in zip(top_ids.tolist(), top_values.tolist(), strict=True):
        rows.append(
            [
                {
                    "logit": float(value),
                    "token": _decode_token(tokenizer, int(token_id)),
                    "token_id": int(token_id),
                }
                for token_id, value in zip(token_ids, values, strict=True)
            ]
        )
    return rows


def _layer_records_for_batch(
    *,
    torch: Any,
    tokenizer: Any,
    output_embedding: Any,
    final_norm: Any | None,
    hidden_states: tuple[Any, ...],
    selected_layers: tuple[int, ...],
    last_positions: list[int],
    format_name: str,
    batch_words: list[str],
    prediction_rows: list[dict[str, Any]],
    token_groups: dict[str, Any],
    top_token_count: int,
) -> list[dict[str, Any]]:
    lens_rows: list[dict[str, Any]] = []
    family_ids: dict[str, list[int]] = token_groups["family_ids"]
    format_ids: dict[str, list[int]] = token_groups["format_ids"]
    for layer in selected_layers:
        hidden = hidden_states[layer]
        row_positions = torch.arange(hidden.shape[0], device=hidden.device)
        last_hidden = hidden[row_positions, last_positions, :]
        normalized = _apply_final_norm(last_hidden, final_norm).float()
        logits = output_embedding(normalized).float()
        log_norm = torch.logsumexp(logits, dim=-1)
        k = max(1, min(int(top_token_count), int(logits.shape[-1])))
        top_values, top_ids = torch.topk(logits, k=k, dim=-1)
        decoded_top_tokens = _top_tokens_per_row(tokenizer=tokenizer, top_ids=top_ids, top_values=top_values)
        family_masses = {
            family: _group_probability_mass(torch, logits, log_norm, ids)
            for family, ids in family_ids.items()
        }
        group_masses = {
            group_name: _group_probability_mass(torch, logits, log_norm, ids)
            for group_name, ids in format_ids.items()
        }
        for row_index, word in enumerate(batch_words):
            prediction_row = prediction_rows[row_index]
            final_family = prediction_row["color_family"]
            available_family_scores = {
                family: float(scores[row_index].item())
                for family, scores in family_masses.items()
                if scores is not None
            }
            best_family = None
            best_family_mass = None
            if available_family_scores:
                best_family, best_family_mass = max(
                    available_family_scores.items(),
                    key=lambda item: item[1],
                )
            matched_family_mass = None
            if final_family is not None and final_family in available_family_scores:
                matched_family_mass = available_family_scores[final_family]
            lens_rows.append(
                {
                    "best_family": best_family,
                    "best_family_mass": best_family_mass,
                    "final_color_family": final_family,
                    "format": format_name,
                    "hex_mass": None
                    if group_masses["hex"] is None
                    else float(group_masses["hex"][row_index].item()),
                    "layer": layer,
                    "matched_family_mass": matched_family_mass,
                    "normalized_output": prediction_row["normalized_output"],
                    "raw_completion": prediction_row["raw_completion"],
                    "rgb_mass": None
                    if group_masses["rgb"] is None
                    else float(group_masses["rgb"][row_index].item()),
                    "temperature": prediction_row["temperature"],
                    "top_tokens": decoded_top_tokens[row_index],
                    "word": word,
                    "word_mass": None
                    if group_masses["word"] is None
                    else float(group_masses["word"][row_index].item()),
                }
            )
    return lens_rows


def _aggregate_layer_summaries(
    *,
    formats: tuple[str, ...],
    layers: tuple[int, ...],
    lens_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layer_summary_rows: list[dict[str, Any]] = []
    top_token_rows: list[dict[str, Any]] = []
    for format_name in formats:
        format_rows = [row for row in lens_rows if row["format"] == format_name]
        for layer in layers:
            subset = [row for row in format_rows if int(row["layer"]) == layer]
            if not subset:
                continue
            accuracy_values = [
                1.0
                for row in subset
                if row["final_color_family"] is not None and row["best_family"] == row["final_color_family"]
            ]
            available_predictions = [
                row for row in subset if row["final_color_family"] is not None and row["best_family"] is not None
            ]
            top1_counts = Counter(
                row["top_tokens"][0]["token"] for row in subset if row.get("top_tokens")
            )
            top_token_rows.append(
                {
                    "format": format_name,
                    "layer": layer,
                    "top_tokens": [
                        {"count": int(count), "token": token}
                        for token, count in top1_counts.most_common(10)
                    ],
                }
            )
            layer_summary_rows.append(
                {
                    "color_family_accuracy": None
                    if not available_predictions
                    else float(len(accuracy_values) / len(available_predictions)),
                    "format": format_name,
                    "layer": layer,
                    "mean_best_family_mass": exp._mean_or_none(
                        [float(row["best_family_mass"]) for row in subset if row["best_family_mass"] is not None]
                    ),
                    "mean_hex_mass": exp._mean_or_none(
                        [float(row["hex_mass"]) for row in subset if row["hex_mass"] is not None]
                    ),
                    "mean_matched_family_mass": exp._mean_or_none(
                        [
                            float(row["matched_family_mass"])
                            for row in subset
                            if row["matched_family_mass"] is not None
                        ]
                    ),
                    "mean_rgb_mass": exp._mean_or_none(
                        [float(row["rgb_mass"]) for row in subset if row["rgb_mass"] is not None]
                    ),
                    "mean_word_mass": exp._mean_or_none(
                        [float(row["word_mass"]) for row in subset if row["word_mass"] is not None]
                    ),
                    "sample_count": len(subset),
                }
            )
    return layer_summary_rows, top_token_rows


def _best_layer_by_metric(
    layer_summary_rows: list[dict[str, Any]],
    *,
    format_name: str,
    metric_name: str,
) -> int | None:
    rows = [
        row
        for row in layer_summary_rows
        if row["format"] == format_name and row.get(metric_name) is not None
    ]
    if not rows:
        return None
    return int(max(rows, key=lambda row: float(row[metric_name]))["layer"])


def _first_rendering_onset_layer(
    layer_summary_rows: list[dict[str, Any]],
    *,
    format_name: str,
    format_metric_name: str,
) -> int | None:
    rows = sorted(
        [
            row
            for row in layer_summary_rows
            if row["format"] == format_name
            and row.get(format_metric_name) is not None
            and row.get("mean_word_mass") is not None
        ],
        key=lambda row: int(row["layer"]),
    )
    for row in rows:
        if float(row[format_metric_name]) > float(row["mean_word_mass"]):
            return int(row["layer"])
    return None


def _write_logit_lens_curve_svg(
    path: Path,
    *,
    layer_summary_rows: list[dict[str, Any]],
    formats: tuple[str, ...],
) -> None:
    rows = [row for row in layer_summary_rows if row.get("sample_count", 0) > 0]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    max_layer = max(int(row["layer"]) for row in rows)
    min_layer = min(int(row["layer"]) for row in rows)
    if min_layer == max_layer:
        max_layer += 1
    panel_width = 360.0
    panel_height = 280.0
    margin = 34.0
    columns = min(3, max(1, len(formats)))
    row_count = math.ceil(len(formats) / columns)
    width = columns * panel_width + 2.0 * margin
    height = row_count * panel_height + 2.0 * margin + 32.0
    inner_left = 48.0
    inner_top = 46.0
    inner_right = 28.0
    inner_bottom = 42.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" viewBox="0 0 {int(width)} {int(height)}">',
        '<rect width="100%" height="100%" fill="#fbfaf7" />',
        f'<text x="{margin}" y="28" font-family="Helvetica, Arial, sans-serif" font-size="22" fill="#111111">Logit lens group mass across layers</text>',
    ]
    metric_specs = [
        ("mean_matched_family_mass", "matched family", GROUP_COLORS["matched_family_mass"]),
        ("mean_word_mass", "word tokens", GROUP_COLORS["word_mass"]),
        ("mean_hex_mass", "hex prefixes", GROUP_COLORS["hex_mass"]),
        ("mean_rgb_mass", "rgb prefixes", GROUP_COLORS["rgb_mass"]),
        ("color_family_accuracy", "family accuracy", GROUP_COLORS["family_accuracy"]),
    ]
    for index, format_name in enumerate(formats):
        panel_x = margin + (index % columns) * panel_width
        panel_y = margin + 10.0 + (index // columns) * panel_height
        inner_width = panel_width - inner_left - inner_right
        inner_height = panel_height - inner_top - inner_bottom
        format_rows = sorted(
            [row for row in rows if row["format"] == format_name],
            key=lambda row: int(row["layer"]),
        )
        lines.extend(
            [
                f'<rect x="{panel_x}" y="{panel_y}" width="{panel_width - 10.0}" height="{panel_height - 10.0}" rx="8" fill="#ffffff" stroke="#d0cbc2" stroke-width="1" />',
                f'<text x="{panel_x + 16.0}" y="{panel_y + 24.0}" font-family="Helvetica, Arial, sans-serif" font-size="16" fill="#111111">{html.escape(format_name)} prompts</text>',
                f'<rect x="{panel_x + inner_left}" y="{panel_y + inner_top}" width="{inner_width}" height="{inner_height}" fill="none" stroke="#bbb5ab" stroke-width="1" />',
            ]
        )

        def project_x(layer: float) -> float:
            return panel_x + inner_left + ((layer - min_layer) / (max_layer - min_layer)) * inner_width

        def project_y(value: float) -> float:
            return panel_y + inner_top + inner_height - value * inner_height

        for guide in (0.2, 0.4, 0.6, 0.8):
            y = project_y(guide)
            lines.append(
                f'<line x1="{panel_x + inner_left}" y1="{y:.2f}" x2="{panel_x + inner_left + inner_width}" y2="{y:.2f}" stroke="#ebe7df" stroke-width="1" />'
            )
        for metric_name, _label, color in metric_specs:
            points = [
                (project_x(float(row["layer"])), project_y(float(row[metric_name])))
                for row in format_rows
                if row.get(metric_name) is not None
            ]
            if len(points) >= 2:
                path_data = " ".join(
                    f"{'M' if point_index == 0 else 'L'} {x:.2f} {y:.2f}"
                    for point_index, (x, y) in enumerate(points)
                )
                dash = ' stroke-dasharray="5 4"' if metric_name == "color_family_accuracy" else ""
                lines.append(
                    f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2.4"{dash} />'
                )
            for x, y in points:
                lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.8" fill="{color}" />')
        for row in format_rows:
            x = project_x(float(row["layer"]))
            lines.append(
                f'<text x="{x - 6.0:.2f}" y="{panel_y + inner_top + inner_height + 18.0:.2f}" font-family="Helvetica, Arial, sans-serif" font-size="10" fill="#666666">{row["layer"]}</text>'
            )
    legend_x = width - 240.0
    legend_y = 34.0
    lines.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="198" height="104" rx="8" fill="#ffffff" stroke="#d0cbc2" stroke-width="1" opacity="0.95" />'
    )
    for index, (_metric_name, label, color) in enumerate(metric_specs):
        y = legend_y + 22.0 + index * 16.0
        dash = ' stroke-dasharray="5 4"' if label == "family accuracy" else ""
        lines.append(
            f'<line x1="{legend_x + 12.0}" y1="{y:.2f}" x2="{legend_x + 32.0}" y2="{y:.2f}" stroke="{color}" stroke-width="2.4"{dash} />'
        )
        lines.append(
            f'<text x="{legend_x + 40.0}" y="{y + 4.0:.2f}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#333333">{html.escape(label)}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_logit_lens_report(
    path: Path,
    *,
    model_name: str,
    word_count: int,
    word_source: str,
    formats: tuple[str, ...],
    parsed_counts: dict[str, int],
    best_accuracy_layers: dict[str, int | None],
    onset_layers: dict[str, int | None],
    interpretation: dict[str, Any],
) -> None:
    lines = [
        "# Cross-format color logit lens report",
        "",
        f"- Model: `{model_name}`",
        f"- Words: `{word_count}` from `{word_source}`",
        f"- Formats: `{', '.join(formats)}`",
        "",
        "## Parse coverage",
        "",
    ]
    for format_name in formats:
        lines.append(f"- `{format_name}` parsed families: `{parsed_counts.get(format_name, 0)}`")
    lines.extend(["", "## Layer summary", ""])
    for format_name in formats:
        best_layer = best_accuracy_layers.get(format_name)
        onset_layer = onset_layers.get(format_name)
        lines.append(
            f"- `{format_name}` best color-family layer: `{ 'n/a' if best_layer is None else best_layer }`"
        )
        if format_name in {"hex", "rgb"}:
            lines.append(
                f"- `{format_name}` rendering-onset layer: `{ 'n/a' if onset_layer is None else onset_layer }`"
            )
    lines.extend(["", "## Interpretation", ""])
    for finding in interpretation.get("headline_findings", []):
        lines.append(f"- {finding}")
    lines.extend(
        [
            "",
            "Artifacts to inspect first:",
            "",
            "- `logit_lens_curve.svg`",
            "- `layer_summary.jsonl`",
            "- `top_token_snapshots.jsonl`",
            "- `logit_lens_rows.jsonl`",
            "- `interpretation.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_logit_lens_final_results(
    output_dir: Path,
    *,
    summary: dict[str, Any],
    interpretation: dict[str, Any],
) -> None:
    payload = {
        "kind": "logit_lens_run",
        "key_artifacts": {
            "curve": "logit_lens_curve.svg",
            "heartbeat_status": "heartbeat_status.json",
            "interpretation": "interpretation.json",
            "interpretation_markdown": "interpretation.md",
            "layer_summary": "layer_summary.jsonl",
            "report": "report.md",
            "rows": "logit_lens_rows.jsonl",
            "summary": "summary.json",
            "top_token_snapshots": "top_token_snapshots.jsonl",
        },
        "interpretation": interpretation,
        "summary": summary,
    }
    exp._write_json(output_dir / "final_results.json", payload)


def _rows_for_format(
    layer_summary_rows: list[dict[str, Any]],
    *,
    format_name: str,
) -> list[dict[str, Any]]:
    return sorted(
        [row for row in layer_summary_rows if row["format"] == format_name],
        key=lambda row: int(row["layer"]),
    )


def _metric_peak(
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
) -> float | None:
    values = [float(row[metric_name]) for row in rows if row.get(metric_name) is not None]
    if not values:
        return None
    return max(values)


def _semantic_color_onset_layer(
    layer_summary_rows: list[dict[str, Any]],
    *,
    format_name: str,
) -> int | None:
    rows = _rows_for_format(layer_summary_rows, format_name=format_name)
    peak_accuracy = _metric_peak(rows, metric_name="color_family_accuracy")
    if peak_accuracy is None:
        return None
    threshold = max(SEMANTIC_ONSET_MIN_ACCURACY, peak_accuracy * SEMANTIC_ONSET_FRACTION)
    for row in rows:
        accuracy = row.get("color_family_accuracy")
        if accuracy is None:
            continue
        if float(accuracy) >= threshold:
            return int(row["layer"])
    return None


def summarize_logit_lens_layers(
    *,
    layer_summary_rows: list[dict[str, Any]],
    formats: tuple[str, ...],
    model_name: str | None = None,
    word_count: int | None = None,
) -> dict[str, Any]:
    best_color_family_layer_by_format = {
        format_name: _best_layer_by_metric(
            layer_summary_rows,
            format_name=format_name,
            metric_name="color_family_accuracy",
        )
        for format_name in formats
    }
    semantic_color_onset_by_format = {
        format_name: _semantic_color_onset_layer(layer_summary_rows, format_name=format_name)
        for format_name in formats
    }
    rendering_onset_layers = {
        "hex": _first_rendering_onset_layer(
            layer_summary_rows,
            format_name="hex",
            format_metric_name="mean_hex_mass",
        )
        if "hex" in formats
        else None,
        "rgb": _first_rendering_onset_layer(
            layer_summary_rows,
            format_name="rgb",
            format_metric_name="mean_rgb_mass",
        )
        if "rgb" in formats
        else None,
    }
    semantic_to_rendering_gap_by_format = {
        format_name: None
        if semantic_color_onset_by_format.get(format_name) is None
        or rendering_onset_layers.get(format_name) is None
        else int(rendering_onset_layers[format_name] - semantic_color_onset_by_format[format_name])
        for format_name in ("hex", "rgb")
        if format_name in formats
    }
    headline_findings: list[str] = []
    if "hex" in formats:
        semantic_layer = semantic_color_onset_by_format.get("hex")
        rendering_layer = rendering_onset_layers.get("hex")
        if semantic_layer is not None and rendering_layer is not None:
            headline_findings.append(
                f"For `hex` prompts, semantic color reaches near-peak accuracy by layer `{semantic_layer}` and hex formatting overtakes color-word mass at layer `{rendering_layer}`."
            )
    if "rgb" in formats:
        semantic_layer = semantic_color_onset_by_format.get("rgb")
        rendering_layer = rendering_onset_layers.get("rgb")
        if semantic_layer is not None and rendering_layer is not None:
            headline_findings.append(
                f"For `rgb` prompts, semantic color reaches near-peak accuracy by layer `{semantic_layer}` and RGB formatting overtakes color-word mass at layer `{rendering_layer}`."
            )
    if "word" in formats:
        word_layer = best_color_family_layer_by_format.get("word")
        if word_layer is not None:
            headline_findings.append(
                f"For `word` prompts, the strongest color-family readout is at layer `{word_layer}`."
            )
    return {
        "best_color_family_layer_by_format": best_color_family_layer_by_format,
        "headline_findings": headline_findings,
        "model_name": model_name,
        "rendering_onset_layers": rendering_onset_layers,
        "semantic_color_onset_by_format": semantic_color_onset_by_format,
        "semantic_to_rendering_gap_by_format": semantic_to_rendering_gap_by_format,
        "word_count": word_count,
    }


def _write_logit_lens_interpretation_markdown(path: Path, *, interpretation: dict[str, Any]) -> None:
    lines = [
        "# Logit lens interpretation",
        "",
    ]
    for finding in interpretation.get("headline_findings", []):
        lines.append(f"- {finding}")
    lines.extend(
        [
            "",
            "## Derived layers",
            "",
            f"- Semantic color onset by format: `{interpretation['semantic_color_onset_by_format']}`",
            f"- Rendering onset layers: `{interpretation['rendering_onset_layers']}`",
            f"- Semantic-to-rendering gaps: `{interpretation['semantic_to_rendering_gap_by_format']}`",
            f"- Best color-family layer by format: `{interpretation['best_color_family_layer_by_format']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_logit_lens_run(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    layer_summary_path = run_dir / "layer_summary.jsonl"
    summary = exp._read_json(summary_path)
    layer_summary_rows = exp._read_prediction_rows(layer_summary_path)
    formats = tuple(str(value) for value in summary.get("formats", exp.DEFAULT_FORMATS))
    interpretation = summarize_logit_lens_layers(
        layer_summary_rows=layer_summary_rows,
        formats=formats,
        model_name=summary.get("model_name"),
        word_count=summary.get("word_count"),
    )
    exp._write_json(run_dir / "interpretation.json", interpretation)
    _write_logit_lens_interpretation_markdown(
        run_dir / "interpretation.md",
        interpretation=interpretation,
    )
    final_results_path = run_dir / "final_results.json"
    if final_results_path.exists():
        payload = exp._read_json(final_results_path)
        payload["interpretation"] = interpretation
        key_artifacts = dict(payload.get("key_artifacts", {}))
        key_artifacts["interpretation"] = "interpretation.json"
        key_artifacts["interpretation_markdown"] = "interpretation.md"
        payload["key_artifacts"] = key_artifacts
        exp._write_json(final_results_path, payload)
    return interpretation


def run_color_logit_lens_experiment(
    *,
    output_dir: Path,
    model_name: str,
    word_list_path: Path | None = None,
    limit: int | None = 200,
    formats: tuple[str, ...] = exp.DEFAULT_FORMATS,
    layers: tuple[int, ...] | None = None,
    max_length: int = 96,
    max_new_tokens: int = 12,
    batch_size: int = 16,
    top_token_count: int = 5,
    resume: bool = False,
    device: str = "auto",
) -> dict[str, Any]:
    np, torch, _PCA, _LogisticRegression, _StratifiedKFold = exp._require_stack()
    output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = exp.HeartbeatRecorder(output_dir, label="logit-lens")
    heartbeat.write_manifest(
        batch_size=batch_size,
        command="logit-lens",
        device=device,
        formats=list(formats),
        limit=limit,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        model_name=model_name,
        requested_layers=None if layers is None else list(layers),
        resume=resume,
        top_token_count=top_token_count,
        word_list_path=None if word_list_path is None else str(word_list_path),
    )
    phase = "setup"
    try:
        heartbeat.update(phase=phase, message="Loading word list")
        words, word_source = exp._read_words(word_list_path, limit)
        state = exp._ensure_checkpoint_state(
            output_dir=output_dir,
            name="logit_lens",
            config={
                "batch_size": batch_size,
                "formats": list(formats),
                "limit": limit,
                "max_length": max_length,
                "max_new_tokens": max_new_tokens,
                "model_name": model_name,
                "requested_layers": None if layers is None else list(layers),
                "top_token_count": top_token_count,
                "word_count": len(words),
                "word_hash": exp._hash_words(words),
            },
            resume=resume,
        )
        tokenizer, model = exp.create_generation_components(model_name)
        device_obj = exp._resolve_device(torch, device)
        model = model.to(device_obj)
        model.eval()
        output_embedding = _get_output_embedding(model)
        final_norm = _find_final_norm(model)
        token_groups = _build_token_groups(tokenizer)
        if not any(token_groups["format_ids"].values()):
            raise RuntimeError("Could not resolve any token groups for logit lens analysis.")
        predictions_by_format: dict[str, list[dict[str, Any]]] = {}
        lens_rows: list[dict[str, Any]] = []
        parsed_counts: dict[str, int] = {}
        selected_layers = (
            None
            if state.get("selected_layers") is None
            else tuple(int(value) for value in state["selected_layers"])
        )

        for format_position, format_name in enumerate(formats, start=1):
            if format_name not in exp.FORMAT_PROMPTS:
                raise ValueError(f"Unsupported format {format_name!r}")
            phase = "collect"
            total_batches = math.ceil(len(words) / batch_size)
            heartbeat.update(
                phase=phase,
                message=f"Collecting logit lens rows for {format_name}",
                current_format=format_name,
                completed_formats=format_position - 1,
                total_formats=len(formats),
            )
            predictions_path = output_dir / f"predictions_{format_name}.jsonl"
            format_complete = (
                resume
                and format_name in set(state.get("completed_formats", []))
                and predictions_path.exists()
            )
            if format_complete:
                format_rows = exp._read_prediction_rows(predictions_path)
                predictions_by_format[format_name] = format_rows
                parsed_counts[format_name] = sum(
                    1 for row in format_rows if row["color_family"] is not None
                )
                loaded_rows: list[dict[str, Any]] = []
                for batch_index in range(1, total_batches + 1):
                    loaded_batch = _load_logit_lens_batch_checkpoint(
                        output_dir=output_dir,
                        format_name=format_name,
                        batch_index=batch_index,
                    )
                    if loaded_batch is None:
                        continue
                    _batch_predictions, batch_rows = loaded_batch
                    loaded_rows.extend(batch_rows)
                if not loaded_rows and (output_dir / "logit_lens_rows.jsonl").exists():
                    loaded_rows = [
                        row
                        for row in exp._read_prediction_rows(output_dir / "logit_lens_rows.jsonl")
                        if row["format"] == format_name
                    ]
                lens_rows.extend(loaded_rows)
                heartbeat.update(
                    phase=phase,
                    message=f"Loaded completed checkpoint for {format_name}",
                    current_format=format_name,
                    parsed_count=parsed_counts[format_name],
                )
                continue

            prompt_template = exp.FORMAT_PROMPTS[format_name]
            format_prediction_rows: list[dict[str, Any]] = []
            for batch_index, start in enumerate(range(0, len(words), batch_size), start=1):
                loaded_checkpoint = (
                    _load_logit_lens_batch_checkpoint(
                        output_dir=output_dir,
                        format_name=format_name,
                        batch_index=batch_index,
                    )
                    if resume
                    else None
                )
                if loaded_checkpoint is not None:
                    batch_predictions, batch_rows = loaded_checkpoint
                    format_prediction_rows.extend(batch_predictions)
                    lens_rows.extend(batch_rows)
                    parsed_counts[format_name] = sum(
                        1 for row in format_prediction_rows if row["color_family"] is not None
                    )
                    if selected_layers is None and batch_rows:
                        selected_layers = tuple(
                            sorted({int(row["layer"]) for row in batch_rows})
                        )
                    heartbeat.update(
                        phase=phase,
                        message=f"Loaded {format_name} batch {batch_index}/{total_batches} from checkpoint",
                        current_format=format_name,
                        processed_words=min(start + batch_size, len(words)),
                        total_words=len(words),
                        batch_index=batch_index,
                        total_batches=total_batches,
                    )
                    continue
                batch_words = words[start : start + batch_size]
                prompts = [
                    exp._render_prompt(tokenizer, prompt_template.format(word=word))
                    for word in batch_words
                ]
                encoded = tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded_device = exp._move_batch_to_device(encoded, device_obj)
                with torch.no_grad():
                    outputs = model(**encoded_device)
                if selected_layers is None:
                    selected_layers = exp._select_layers(outputs.hidden_states, layers)
                    state["selected_layers"] = list(selected_layers)
                    exp._save_checkpoint_state(output_dir, "logit_lens", state)
                last_positions = exp._non_padding_last_positions(encoded_device["attention_mask"])
                generation_kwargs: dict[str, Any] = {
                    **encoded_device,
                    "do_sample": False,
                    "max_new_tokens": max_new_tokens,
                }
                pad_token_id = getattr(tokenizer, "pad_token_id", None)
                eos_token_id = getattr(tokenizer, "eos_token_id", None)
                if pad_token_id is not None:
                    generation_kwargs["pad_token_id"] = pad_token_id
                if eos_token_id is not None:
                    generation_kwargs["eos_token_id"] = eos_token_id
                with torch.no_grad():
                    generated = model.generate(**generation_kwargs)
                prompt_length = int(encoded_device["input_ids"].shape[1])
                completions = tokenizer.batch_decode(
                    generated[:, prompt_length:].detach().cpu(),
                    skip_special_tokens=True,
                )
                batch_predictions: list[dict[str, Any]] = []
                for word, prompt, raw_completion in zip(batch_words, prompts, completions, strict=True):
                    parsed = exp.parse_format_completion(format_name, raw_completion)
                    batch_predictions.append(
                        {
                            "color_family": parsed.color_family,
                            "format": format_name,
                            "normalized_output": parsed.normalized_output,
                            "parse_success": parsed.color_family is not None,
                            "prompt": prompt,
                            "raw_completion": raw_completion.strip(),
                            "temperature": parsed.temperature,
                            "word": word,
                        }
                    )
                batch_rows = _layer_records_for_batch(
                    torch=torch,
                    tokenizer=tokenizer,
                    output_embedding=output_embedding,
                    final_norm=final_norm,
                    hidden_states=outputs.hidden_states,
                    selected_layers=selected_layers,
                    last_positions=last_positions,
                    format_name=format_name,
                    batch_words=batch_words,
                    prediction_rows=batch_predictions,
                    token_groups=token_groups,
                    top_token_count=top_token_count,
                )
                _save_logit_lens_batch_checkpoint(
                    output_dir=output_dir,
                    format_name=format_name,
                    batch_index=batch_index,
                    prediction_rows=batch_predictions,
                    lens_rows=batch_rows,
                )
                format_prediction_rows.extend(batch_predictions)
                lens_rows.extend(batch_rows)
                parsed_counts[format_name] = sum(
                    1 for row in format_prediction_rows if row["color_family"] is not None
                )
                heartbeat.update(
                    phase=phase,
                    message=f"Completed {format_name} batch {batch_index}/{total_batches}",
                    batch_index=batch_index,
                    current_format=format_name,
                    parsed_count=parsed_counts[format_name],
                    processed_words=min(start + len(batch_words), len(words)),
                    total_batches=total_batches,
                    total_words=len(words),
                )
            predictions_by_format[format_name] = format_prediction_rows
            exp._write_jsonl(predictions_path, format_prediction_rows)
            completed_formats = set(state.get("completed_formats", []))
            completed_formats.add(format_name)
            state["completed_formats"] = sorted(completed_formats)
            exp._save_checkpoint_state(output_dir, "logit_lens", state)
            heartbeat.event(
                phase=phase,
                message=f"Saved logit lens predictions for {format_name}",
                current_format=format_name,
                parsed_count=parsed_counts[format_name],
            )

        if selected_layers is None:
            raise RuntimeError("No hidden states were collected for logit lens.")

        phase = "analyze"
        heartbeat.update(phase=phase, message="Aggregating logit lens summaries")
        exp._write_jsonl(output_dir / "logit_lens_rows.jsonl", lens_rows)
        layer_summary_rows, top_token_rows = _aggregate_layer_summaries(
            formats=formats,
            layers=selected_layers,
            lens_rows=lens_rows,
        )
        exp._write_jsonl(output_dir / "layer_summary.jsonl", layer_summary_rows)
        exp._write_jsonl(output_dir / "top_token_snapshots.jsonl", top_token_rows)
        _write_logit_lens_curve_svg(
            output_dir / "logit_lens_curve.svg",
            layer_summary_rows=layer_summary_rows,
            formats=formats,
        )
        interpretation = summarize_logit_lens_layers(
            layer_summary_rows=layer_summary_rows,
            formats=formats,
            model_name=model_name,
            word_count=len(words),
        )
        exp._write_json(output_dir / "interpretation.json", interpretation)
        _write_logit_lens_interpretation_markdown(
            output_dir / "interpretation.md",
            interpretation=interpretation,
        )
        summary = {
            "best_color_family_layer_by_format": interpretation["best_color_family_layer_by_format"],
            "formats": list(formats),
            "interpretation": interpretation,
            "layers": list(selected_layers),
            "model_name": model_name,
            "parsed_counts_by_format": parsed_counts,
            "rendering_onset_layers": interpretation["rendering_onset_layers"],
            "resume": resume,
            "semantic_color_onset_by_format": interpretation["semantic_color_onset_by_format"],
            "semantic_to_rendering_gap_by_format": interpretation["semantic_to_rendering_gap_by_format"],
            "top_token_count": top_token_count,
            "word_count": len(words),
            "word_source": word_source,
        }
        exp._write_json(output_dir / "summary.json", summary)
        _write_logit_lens_report(
            output_dir / "report.md",
            model_name=model_name,
            word_count=len(words),
            word_source=word_source,
            formats=formats,
            parsed_counts=parsed_counts,
            best_accuracy_layers=interpretation["best_color_family_layer_by_format"],
            onset_layers=interpretation["rendering_onset_layers"],
            interpretation=interpretation,
        )
        _write_logit_lens_final_results(
            output_dir,
            summary=summary,
            interpretation=interpretation,
        )
        heartbeat.update(
            phase="completed",
            message="Logit lens run complete",
            state="completed",
            best_color_family_layer_by_format=interpretation["best_color_family_layer_by_format"],
            rendering_onset_layers=interpretation["rendering_onset_layers"],
        )
        return summary
    except Exception as error:
        heartbeat.fail(phase=phase, error=error)
        raise
