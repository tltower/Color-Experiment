from __future__ import annotations

from pathlib import Path
from typing import Any

from .analysis_common import cosine_similarity_matrix
from .model_utils import _resolve_device
from .run_support import HeartbeatRecorder, _write_json, _write_jsonl
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
    "run_word_set_sae_feature_experiment",
]
