from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .analysis_common import parse_layers as _parse_layers
from .color_formats import (
    DEFAULT_FORMATS,
    FORMAT_PROMPTS,
    parse_format_completion,
)
from .format_analysis import _build_layer_analysis, _fit_transfer_accuracy as _shared_fit_transfer_accuracy, _mean_or_none
from .hf import create_generation_components
from .model_utils import (
    _coerce_hidden_output,
    _find_transformer_blocks,
    _move_batch_to_device,
    _non_padding_last_positions,
    _render_prompt,
    _resolve_device,
)
from .run_support import (
    HeartbeatRecorder,
    _ensure_checkpoint_state,
    _hash_words,
    _read_json,
    _read_prediction_rows,
    _save_checkpoint_state,
    _utc_now,
    _write_json,
    _write_jsonl,
)
from .word_lists import (
    WORD_PRESET_NAMES,
    bundled_color_word_list_path,
    default_words,
    find_system_word_list,
    preset_words,
    read_word_file,
)
from .workflow_common import _require_stack, _select_layers


def _format_checkpoint_dir(output_dir: Path, format_name: str) -> Path:
    return output_dir / "checkpoints" / "run_batches" / format_name


def _format_batch_predictions_path(output_dir: Path, format_name: str, batch_index: int) -> Path:
    return _format_checkpoint_dir(output_dir, format_name) / f"batch_{batch_index:04d}.predictions.jsonl"


def _format_batch_hidden_path(output_dir: Path, format_name: str, batch_index: int) -> Path:
    return _format_checkpoint_dir(output_dir, format_name) / f"batch_{batch_index:04d}.hidden_states.npz"


def _save_format_batch_checkpoint(
    *,
    np: Any,
    output_dir: Path,
    format_name: str,
    batch_index: int,
    rows: list[dict[str, Any]],
    layer_arrays: dict[int, Any],
) -> None:
    checkpoint_dir = _format_checkpoint_dir(output_dir, format_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(_format_batch_predictions_path(output_dir, format_name, batch_index), rows)
    np.savez_compressed(
        _format_batch_hidden_path(output_dir, format_name, batch_index),
        **{f"layer_{layer:02d}": array.astype(np.float16) for layer, array in layer_arrays.items()},
    )


def _load_format_batch_checkpoint(
    *,
    np: Any,
    output_dir: Path,
    format_name: str,
    batch_index: int,
) -> tuple[list[dict[str, Any]], dict[int, Any]] | None:
    predictions_path = _format_batch_predictions_path(output_dir, format_name, batch_index)
    hidden_path = _format_batch_hidden_path(output_dir, format_name, batch_index)
    if not predictions_path.exists() or not hidden_path.exists():
        return None
    rows = _read_prediction_rows(predictions_path)
    hidden_arrays: dict[int, Any] = {}
    with np.load(hidden_path) as payload:
        for key in payload.files:
            hidden_arrays[int(key.rsplit("_", 1)[1])] = payload[key].astype(np.float32)
    return rows, hidden_arrays


def _patch_checkpoint_dir(output_dir: Path) -> Path:
    return output_dir / "checkpoints" / "patch_batches"


def _patch_batch_path(output_dir: Path, batch_index: int) -> Path:
    return _patch_checkpoint_dir(output_dir) / f"batch_{batch_index:04d}.jsonl"


def _save_patch_batch_checkpoint(output_dir: Path, batch_index: int, rows: list[dict[str, Any]]) -> None:
    checkpoint_dir = _patch_checkpoint_dir(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(_patch_batch_path(output_dir, batch_index), rows)


def _load_patch_batch_checkpoint(output_dir: Path, batch_index: int) -> list[dict[str, Any]] | None:
    path = _patch_batch_path(output_dir, batch_index)
    if not path.exists():
        return None
    return _read_prediction_rows(path)


def _read_words(
    word_list_path: Path | None,
    limit: int | None,
    *,
    word_preset: str = "default",
) -> tuple[list[str], str]:
    if word_list_path is not None:
        source = str(word_list_path)
        words = read_word_file(word_list_path, limit=limit)
    elif word_preset == "color_words":
        source = "color_words"
        words = preset_words("color_words", limit=limit)
    else:
        default_seed = default_words()
        if limit is None or limit <= len(default_seed):
            source = word_preset
            words = preset_words(word_preset, limit=limit)
        else:
            system_dictionary = find_system_word_list()
            if system_dictionary is None:
                raise RuntimeError(
                    f"Requested {limit} words, but only {len(default_seed)} built-in words are available. "
                    "Install a system dictionary or pass `--word-list-path`."
                )
            source = str(system_dictionary)
            words = read_word_file(system_dictionary, limit=limit)
    if not words:
        raise ValueError("Word list resolved to zero words.")
    return words, source


def _fit_transfer_accuracy(*args: Any, **kwargs: Any) -> float | None:
    return _shared_fit_transfer_accuracy(*args, **kwargs)


def _derive_consensus_labels(
    predictions_by_format: dict[str, list[dict[str, Any]]],
    *,
    min_votes: int,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    word_to_families: dict[str, list[str]] = {}
    for rows in predictions_by_format.values():
        for row in rows:
            family = row.get("color_family")
            if family is None:
                continue
            word_to_families.setdefault(row["word"], []).append(family)

    consensus_by_word: dict[str, str] = {}
    consensus_rows: list[dict[str, Any]] = []
    for word in sorted(word_to_families):
        votes = Counter(word_to_families[word])
        top_two = votes.most_common(2)
        if not top_two or top_two[0][1] < min_votes:
            continue
        if len(top_two) > 1 and top_two[0][1] == top_two[1][1]:
            continue
        family = top_two[0][0]
        consensus_by_word[word] = family
        consensus_rows.append(
            {
                "consensus_color_family": family,
                "vote_count": top_two[0][1],
                "vote_distribution": dict(votes),
                "word": word,
            }
        )
    return consensus_by_word, consensus_rows


def _write_run_report(
    path: Path,
    *,
    model_name: str,
    word_count: int,
    word_source: str,
    formats: tuple[str, ...],
    parsed_counts: dict[str, int],
    consensus_count: int,
    best_cross_layer: int | None,
    best_cross_mean_accuracy: float | None,
    best_within_schema_accuracy_by_format: dict[str, float | None],
    best_within_schema_layer_by_format: dict[str, int | None],
) -> None:
    accuracy_text = "n/a" if best_cross_mean_accuracy is None else f"{best_cross_mean_accuracy:.3f}"
    layer_text = "n/a" if best_cross_layer is None else str(best_cross_layer)
    lines = [
        "# Color latent report",
        "",
        f"- Model: `{model_name}`",
        f"- Words: `{word_count}` from `{word_source}`",
        f"- Formats: `{', '.join(formats)}`",
        f"- Consensus-labelled words: `{consensus_count}`",
        "",
        "## Parse coverage",
        "",
    ]
    for format_name in formats:
        lines.append(f"- `{format_name}` parsed families: `{parsed_counts.get(format_name, 0)}`")
    lines.extend(
        [
            "",
            "## Within-schema probes",
            "",
        ]
    )
    for format_name in formats:
        layer = best_within_schema_layer_by_format.get(format_name)
        accuracy = best_within_schema_accuracy_by_format.get(format_name)
        layer_text_for_format = "n/a" if layer is None else str(layer)
        accuracy_text_for_format = "n/a" if accuracy is None else f"{accuracy:.3f}"
        lines.append(
            f"- `{format_name}` exact-output best layer: `{layer_text_for_format}`; accuracy: `{accuracy_text_for_format}`"
        )
    lines.extend(
        [
            "",
            "## Probe transfer summary",
            "",
            f"- Best cross-format layer: `{layer_text}`",
            f"- Best cross-format mean accuracy: `{accuracy_text}`",
            "",
            "Artifacts to inspect first:",
            "",
            "- `shared_pca_grid.svg`",
            "- `format_transfer_curve.svg`",
            "- `within_schema_probe_accuracy.jsonl`",
            "- `cross_format_probe_transfer.jsonl`",
            "- `shared_pca_points.jsonl`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_run_final_results(
    output_dir: Path,
    *,
    summary: dict[str, Any],
    layer_summary_rows: list[dict[str, Any]],
) -> None:
    top_layers = sorted(
        [row for row in layer_summary_rows if row.get("cross_format_mean") is not None],
        key=lambda row: float(row["cross_format_mean"]),
        reverse=True,
    )[:5]
    payload = {
        "kind": "cross_format_run",
        "key_artifacts": {
            "consensus_labels": "consensus_labels.jsonl",
            "cross_format_probe_transfer": "cross_format_probe_transfer.jsonl",
            "heartbeat_status": "heartbeat_status.json",
            "layer_summary": "layer_summary.jsonl",
            "probe_transfer": "probe_transfer.jsonl",
            "report": "report.md",
            "shared_pca_grid": "shared_pca_grid.svg",
            "summary": "summary.json",
            "transfer_curve": "format_transfer_curve.svg",
            "within_schema_probe_accuracy": "within_schema_probe_accuracy.jsonl",
        },
        "summary": summary,
        "top_cross_layers": top_layers,
    }
    _write_json(output_dir / "final_results.json", payload)


def run_color_format_latent_experiment(
    *,
    output_dir: Path,
    model_name: str,
    word_list_path: Path | None = None,
    word_preset: str = "default",
    limit: int | None = 1000,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    layers: tuple[int, ...] | None = None,
    max_length: int = 96,
    max_new_tokens: int = 12,
    batch_size: int = 16,
    grid_stride: int = 4,
    min_consensus_votes: int = 2,
    resume: bool = False,
    device: str = "auto",
) -> dict[str, Any]:
    np, torch, PCA, LogisticRegression, KFold = _require_stack()
    output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = HeartbeatRecorder(output_dir, label="format-latent-run")
    heartbeat.write_manifest(
        batch_size=batch_size,
        command="run",
        device=device,
        formats=list(formats),
        grid_stride=grid_stride,
        limit=limit,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        min_consensus_votes=min_consensus_votes,
        model_name=model_name,
        requested_layers=None if layers is None else list(layers),
        resume=resume,
        word_list_path=None if word_list_path is None else str(word_list_path),
        word_preset=word_preset,
    )
    phase = "setup"
    try:
        heartbeat.update(phase=phase, message="Loading word list")
        words, word_source = _read_words(word_list_path, limit, word_preset=word_preset)
        state = _ensure_checkpoint_state(
            output_dir=output_dir,
            name="run",
            config={
                "batch_size": batch_size,
                "formats": list(formats),
                "grid_stride": grid_stride,
                "limit": limit,
                "max_length": max_length,
                "max_new_tokens": max_new_tokens,
                "min_consensus_votes": min_consensus_votes,
                "model_name": model_name,
                "requested_layers": None if layers is None else list(layers),
                "word_count": len(words),
                "word_hash": _hash_words(words),
                "word_preset": word_preset,
            },
            resume=resume,
        )
        tokenizer, model = create_generation_components(model_name)
        device_obj = _resolve_device(torch, device)
        model = model.to(device_obj)
        model.eval()
        hidden_root = output_dir / "hidden_states"
        hidden_root.mkdir(parents=True, exist_ok=True)
        predictions_by_format: dict[str, list[dict[str, Any]]] = {}
        parsed_counts: dict[str, int] = {}
        selected_layers = (
            None
            if state.get("selected_layers") is None
            else tuple(int(value) for value in state["selected_layers"])
        )

        for format_position, format_name in enumerate(formats, start=1):
            if format_name not in FORMAT_PROMPTS:
                raise ValueError(f"Unsupported format {format_name!r}")
            phase = "collect"
            heartbeat.update(
                phase=phase,
                message=f"Collecting {format_name} activations",
                current_format=format_name,
                completed_formats=format_position - 1,
                total_formats=len(formats),
            )
            predictions_path = output_dir / f"predictions_{format_name}.jsonl"
            format_hidden_dir = hidden_root / format_name
            format_complete = (
                resume
                and format_name in set(state.get("completed_formats", []))
                and predictions_path.exists()
                and format_hidden_dir.exists()
            )
            if format_complete:
                format_rows = _read_prediction_rows(predictions_path)
                predictions_by_format[format_name] = format_rows
                parsed_counts[format_name] = sum(
                    1 for row in format_rows if row["color_family"] is not None
                )
                heartbeat.update(
                    phase=phase,
                    message=f"Loaded completed checkpoint for {format_name}",
                    current_format=format_name,
                    parsed_count=parsed_counts[format_name],
                )
                continue
            format_rows: list[dict[str, Any]] = []
            layer_chunks: dict[int, list[Any]] = {}
            prompt_template = FORMAT_PROMPTS[format_name]
            total_batches = math.ceil(len(words) / batch_size)
            for batch_index, start in enumerate(range(0, len(words), batch_size), start=1):
                loaded_checkpoint = (
                    _load_format_batch_checkpoint(
                        np=np,
                        output_dir=output_dir,
                        format_name=format_name,
                        batch_index=batch_index,
                    )
                    if resume
                    else None
                )
                if loaded_checkpoint is not None:
                    batch_rows, batch_layer_arrays = loaded_checkpoint
                    format_rows.extend(batch_rows)
                    for layer, batch_array in batch_layer_arrays.items():
                        layer_chunks.setdefault(layer, []).append(batch_array.astype(np.float32))
                    parsed_counts[format_name] = sum(
                        1 for row in format_rows if row["color_family"] is not None
                    )
                    if selected_layers is None:
                        selected_layers = tuple(sorted(batch_layer_arrays))
                    heartbeat.update(
                        phase=phase,
                        message=f"Loaded {format_name} batch {batch_index}/{total_batches} from checkpoint",
                        current_format=format_name,
                        processed_words=min(start + batch_size, len(words)),
                        total_words=len(words),
                        batch_index=batch_index,
                        total_batches=total_batches,
                        parsed_count=parsed_counts[format_name],
                    )
                    continue
                batch_words = words[start : start + batch_size]
                prompts = [
                    _render_prompt(tokenizer, prompt_template.format(word=word))
                    for word in batch_words
                ]
                encoded = tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded_device = _move_batch_to_device(encoded, device_obj)
                with torch.no_grad():
                    outputs = model(**encoded_device)
                if selected_layers is None:
                    selected_layers = _select_layers(outputs.hidden_states, layers)
                    state["selected_layers"] = list(selected_layers)
                    _save_checkpoint_state(output_dir, "run", state)
                last_positions = _non_padding_last_positions(encoded_device["attention_mask"])
                batch_layer_arrays: dict[int, Any] = {}
                for layer in selected_layers:
                    hidden_state = outputs.hidden_states[layer].detach().float().cpu()
                    batch_layer_array = np.stack(
                        [
                            hidden_state[row_index, last_position, :].numpy().astype(np.float32)
                            for row_index, last_position in enumerate(last_positions)
                        ]
                    )
                    batch_layer_arrays[layer] = batch_layer_array
                    layer_chunks.setdefault(layer, []).append(batch_layer_array)

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
                batch_rows: list[dict[str, Any]] = []
                for word, prompt, raw_completion in zip(batch_words, prompts, completions, strict=True):
                    parsed = parse_format_completion(format_name, raw_completion)
                    batch_rows.append(
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
                _save_format_batch_checkpoint(
                    np=np,
                    output_dir=output_dir,
                    format_name=format_name,
                    batch_index=batch_index,
                    rows=batch_rows,
                    layer_arrays=batch_layer_arrays,
                )
                format_rows.extend(batch_rows)
                parsed_counts[format_name] = sum(
                    1 for row in format_rows if row["color_family"] is not None
                )
                heartbeat.update(
                    phase=phase,
                    message=f"Completed {format_name} batch {batch_index}/{total_batches}",
                    current_format=format_name,
                    processed_words=min(start + len(batch_words), len(words)),
                    total_words=len(words),
                    batch_index=batch_index,
                    total_batches=total_batches,
                    parsed_count=parsed_counts[format_name],
                )

            predictions_by_format[format_name] = format_rows
            _write_jsonl(predictions_path, format_rows)
            format_hidden_dir.mkdir(parents=True, exist_ok=True)
            for layer in selected_layers or ():
                layer_array = np.concatenate(layer_chunks[layer], axis=0).astype(np.float16)
                np.save(format_hidden_dir / f"layer_{layer:02d}.npy", layer_array)
            completed_formats = set(state.get("completed_formats", []))
            completed_formats.add(format_name)
            state["completed_formats"] = sorted(completed_formats)
            _save_checkpoint_state(output_dir, "run", state)
            heartbeat.event(
                phase=phase,
                message=f"Saved {format_name} predictions and hidden states",
                current_format=format_name,
                parsed_count=parsed_counts[format_name],
            )

        if selected_layers is None:
            raise RuntimeError("No hidden states were collected.")

        phase = "consensus"
        heartbeat.update(phase=phase, message="Deriving consensus labels")
        consensus_by_word, consensus_rows = _derive_consensus_labels(
            predictions_by_format,
            min_votes=min_consensus_votes,
        )
        _write_jsonl(output_dir / "consensus_labels.jsonl", consensus_rows)

        phase = "analyze"
        analysis_summary = _build_layer_analysis(
            output_dir=output_dir,
            formats=formats,
            layers=selected_layers,
            predictions_by_format=predictions_by_format,
            consensus_by_word=consensus_by_word,
            grid_stride=grid_stride,
            heartbeat=heartbeat,
            np=np,
            PCA=PCA,
            LogisticRegression=LogisticRegression,
            KFold=KFold,
        )

        summary = {
            "best_cross_layer": analysis_summary["best_cross_layer"],
            "best_cross_mean_accuracy": analysis_summary["best_cross_mean_accuracy"],
            "consensus_word_count": len(consensus_by_word),
            "formats": list(formats),
            "layers": list(selected_layers),
            "model_name": model_name,
            "parsed_counts_by_format": parsed_counts,
            "resume": resume,
            "word_count": len(words),
            "word_preset": word_preset,
            "word_source": word_source,
            "best_within_schema_accuracy_by_format": analysis_summary["best_within_schema_accuracy_by_format"],
            "best_within_schema_layer_by_format": analysis_summary["best_within_schema_layer_by_format"],
        }
        _write_json(output_dir / "summary.json", summary)
        _write_run_report(
            output_dir / "report.md",
            model_name=model_name,
            word_count=len(words),
            word_source=word_source,
            formats=formats,
            parsed_counts=parsed_counts,
            consensus_count=len(consensus_by_word),
            best_cross_layer=analysis_summary["best_cross_layer"],
            best_cross_mean_accuracy=analysis_summary["best_cross_mean_accuracy"],
            best_within_schema_accuracy_by_format=analysis_summary["best_within_schema_accuracy_by_format"],
            best_within_schema_layer_by_format=analysis_summary["best_within_schema_layer_by_format"],
        )
        _write_run_final_results(
            output_dir,
            summary=summary,
            layer_summary_rows=analysis_summary["layer_summary_rows"],
        )
        phase = "completed"
        heartbeat.update(
            phase=phase,
            message="Cross-format run complete",
            state="completed",
            best_cross_layer=analysis_summary["best_cross_layer"],
            consensus_word_count=len(consensus_by_word),
            parsed_counts_by_format=parsed_counts,
        )
        return summary
    except Exception as error:
        heartbeat.fail(phase=phase, error=error)
        raise
def _load_pairs(
    *,
    pairs_path: Path | None,
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    limit: int | None,
) -> list[dict[str, str]]:
    source_words = {row["word"] for row in source_rows}
    target_words = {row["word"] for row in target_rows}
    if pairs_path is None:
        pairs = [{"source_word": word, "target_word": word} for word in sorted(source_words & target_words)]
    else:
        pairs = []
        for raw_line in pairs_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if pairs_path.suffix == ".jsonl":
                row = json.loads(line)
                source_word = str(row["source_word"]).strip().lower()
                target_word = str(row["target_word"]).strip().lower()
            else:
                left, right = [chunk.strip().lower() for chunk in line.split(",", 1)]
                source_word = left
                target_word = right
            if source_word in source_words and target_word in target_words:
                pairs.append({"source_word": source_word, "target_word": target_word})
    if limit is not None:
        pairs = pairs[:limit]
    if not pairs:
        raise ValueError("No valid source/target pairs were found for patching.")
    return pairs


def _write_patch_report(
    path: Path,
    *,
    layer: int,
    source_format: str,
    target_format: str,
    pair_count: int,
    changed_rate: float | None,
    moved_toward_source_rate: float | None,
    patched_match_rate: float | None,
) -> None:
    changed_text = "n/a" if changed_rate is None else f"{changed_rate:.3f}"
    moved_text = "n/a" if moved_toward_source_rate is None else f"{moved_toward_source_rate:.3f}"
    patched_text = "n/a" if patched_match_rate is None else f"{patched_match_rate:.3f}"
    lines = [
        "# Cross-format residual patch report",
        "",
        f"- Source format: `{source_format}`",
        f"- Target format: `{target_format}`",
        f"- Layer: `{layer}`",
        f"- Pairs: `{pair_count}`",
        f"- Changed rate: `{changed_text}`",
        f"- Moved toward source rate: `{moved_text}`",
        f"- Patched/source family match rate: `{patched_text}`",
        "",
        "Artifacts to inspect first:",
        "",
        "- `patched_predictions.jsonl`",
        "- `summary.json`",
        "- `heartbeat_status.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_patch_final_results(output_dir: Path, *, summary: dict[str, Any]) -> None:
    payload = {
        "kind": "cross_format_patch",
        "key_artifacts": {
            "heartbeat_status": "heartbeat_status.json",
            "patched_predictions": "patched_predictions.jsonl",
            "report": "report.md",
            "summary": "summary.json",
        },
        "summary": summary,
    }
    _write_json(output_dir / "final_results.json", payload)


def export_final_results(
    *,
    run_dir: Path,
    output_dir: Path,
    patch_dir: Path | None = None,
    logit_lens_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_final_results_path = run_dir / "final_results.json"
    run_payload = _read_json(run_final_results_path) if run_final_results_path.exists() else _read_json(run_dir / "summary.json")
    copied_files: list[dict[str, str]] = []

    def copy_file(source: Path, destination_name: str) -> None:
        if not source.exists():
            return
        destination = output_dir / destination_name
        shutil.copy2(source, destination)
        copied_files.append({"source": str(source), "destination": str(destination)})

    for name in (
        "final_results.json",
        "summary.json",
        "report.md",
        "shared_pca_grid.svg",
        "format_transfer_curve.svg",
        "cross_format_probe_transfer.jsonl",
        "within_schema_probe_accuracy.jsonl",
        "layer_summary.jsonl",
        "probe_transfer.jsonl",
        "heartbeat_status.json",
    ):
        copy_file(run_dir / name, f"run_{name}")

    payload: dict[str, Any] = {
        "copied_files": copied_files,
        "created_at_utc": _utc_now(),
        "logit_lens_dir": None if logit_lens_dir is None else str(logit_lens_dir),
        "patch_dir": None if patch_dir is None else str(patch_dir),
        "run_dir": str(run_dir),
        "run_summary": run_payload.get("summary", run_payload),
    }
    if patch_dir is not None:
        patch_final_results_path = patch_dir / "final_results.json"
        patch_payload = _read_json(patch_final_results_path) if patch_final_results_path.exists() else _read_json(patch_dir / "summary.json")
        payload["patch_summary"] = patch_payload.get("summary", patch_payload)
        for name in (
            "final_results.json",
            "summary.json",
            "report.md",
            "patched_predictions.jsonl",
            "heartbeat_status.json",
        ):
            copy_file(patch_dir / name, f"patch_{name}")
    if logit_lens_dir is not None:
        logit_lens_final_results_path = logit_lens_dir / "final_results.json"
        logit_lens_payload = (
            _read_json(logit_lens_final_results_path)
            if logit_lens_final_results_path.exists()
            else _read_json(logit_lens_dir / "summary.json")
        )
        payload["logit_lens_summary"] = logit_lens_payload.get("summary", logit_lens_payload)
        for name in (
            "final_results.json",
            "summary.json",
            "report.md",
            "interpretation.json",
            "interpretation.md",
            "logit_lens_curve.svg",
            "layer_summary.jsonl",
            "top_token_snapshots.jsonl",
            "heartbeat_status.json",
        ):
            copy_file(logit_lens_dir / name, f"logit_lens_{name}")
    _write_json(output_dir / "final_results_bundle.json", payload)
    return payload


def _load_summary_payload(output_dir: Path) -> dict[str, Any]:
    final_results_path = output_dir / "final_results.json"
    if final_results_path.exists():
        payload = _read_json(final_results_path)
        if isinstance(payload.get("summary"), dict):
            return dict(payload["summary"])
        return payload
    summary_path = output_dir / "summary.json"
    return _read_json(summary_path)


def _write_color_word_basis_report(
    path: Path,
    *,
    model_name: str,
    word_count: int,
    word_source: str,
    run_summary: dict[str, Any],
    logit_lens_summary: dict[str, Any],
    sae_training_summary: dict[str, Any],
    sae_analysis_summary: dict[str, Any],
) -> None:
    lines = [
        "# Color-word basis experiment",
        "",
        f"- Model: `{model_name}`",
        f"- Color words: `{word_count}` from `{word_source}`",
        f"- Best cross-format layer: `{run_summary.get('best_cross_layer')}`",
        f"- Best within-schema layers: `{run_summary.get('best_within_schema_layer_by_format')}`",
        f"- Logit-lens best family layers: `{logit_lens_summary.get('best_color_family_layer_by_format')}`",
        f"- SAE layer: `{sae_training_summary.get('layer')}`",
        f"- SAE dictionary size: `{sae_training_summary.get('dictionary_size')}`",
        "",
        "Artifacts to inspect first:",
        "",
        "- `run/shared_pca_grid.svg`",
        "- `run/within_schema_probe_accuracy.jsonl`",
        "- `run/cross_format_probe_transfer.jsonl`",
        "- `logit_lens/interpretation.md`",
        "- `sae_analysis_word/schema_label_rankings.json`",
        "- `sae_analysis_word/family_rankings.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_color_word_basis_final_results(output_dir: Path, *, summary: dict[str, Any]) -> None:
    payload = {
        "kind": "color_word_basis_experiment",
        "key_artifacts": {
            "heartbeat_status": "heartbeat_status.json",
            "report": "report.md",
            "run_summary": "run/summary.json",
            "logit_lens_summary": "logit_lens/summary.json",
            "sae_training_summary": "sae_train_word/summary.json",
            "sae_analysis_summary": "sae_analysis_word/summary.json",
        },
        "summary": summary,
    }
    _write_json(output_dir / "final_results.json", payload)


def run_color_word_basis_experiment(
    *,
    output_dir: Path,
    model_name: str,
    word_list_path: Path | None = None,
    limit: int | None = None,
    layers: tuple[int, ...] | None = None,
    max_length: int = 96,
    max_new_tokens: int = 12,
    batch_size: int = 16,
    grid_stride: int = 4,
    min_consensus_votes: int = 2,
    logit_lens_top_token_count: int = 8,
    sae_layer: int = 16,
    sae_prompt_format: str = "word",
    sae_max_length: int = 256,
    sae_activation_batch_size: int = 32,
    sae_train_batch_size: int = 256,
    sae_dictionary_multiplier: int = 8,
    sae_dictionary_size: int | None = None,
    sae_top_k: int | None = 32,
    sae_epochs: int = 8,
    sae_learning_rate: float = 1e-3,
    sae_l1_coefficient: float = 1e-4,
    sae_validation_fraction: float = 0.1,
    sae_seed: int = 17,
    resume: bool = False,
    device: str = "auto",
) -> dict[str, Any]:
    from .custom_sae import run_color_sae_feature_analysis, run_color_sae_training
    from .logit_lens import run_color_logit_lens_experiment

    output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = HeartbeatRecorder(output_dir, label="color-word-basis")
    resolved_word_list_path = word_list_path
    if resolved_word_list_path is None:
        candidate = bundled_color_word_list_path()
        if candidate.exists():
            resolved_word_list_path = candidate
    word_preset = "default" if resolved_word_list_path is not None else "color_words"
    preview_words, word_source = _read_words(
        resolved_word_list_path,
        limit,
        word_preset=word_preset,
    )
    resolved_limit = len(preview_words) if limit is None else limit
    heartbeat.write_manifest(
        batch_size=batch_size,
        command="color-word-basis",
        device=device,
        grid_stride=grid_stride,
        layers=None if layers is None else list(layers),
        limit=resolved_limit,
        logit_lens_top_token_count=logit_lens_top_token_count,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        min_consensus_votes=min_consensus_votes,
        model_name=model_name,
        resume=resume,
        sae_activation_batch_size=sae_activation_batch_size,
        sae_dictionary_multiplier=sae_dictionary_multiplier,
        sae_dictionary_size=sae_dictionary_size,
        sae_epochs=sae_epochs,
        sae_l1_coefficient=sae_l1_coefficient,
        sae_layer=sae_layer,
        sae_learning_rate=sae_learning_rate,
        sae_max_length=sae_max_length,
        sae_prompt_format=sae_prompt_format,
        sae_top_k=sae_top_k,
        sae_train_batch_size=sae_train_batch_size,
        sae_validation_fraction=sae_validation_fraction,
        word_count=len(preview_words),
        word_list_path=None if resolved_word_list_path is None else str(resolved_word_list_path),
        word_source=word_source,
    )
    phase = "setup"
    try:
        heartbeat.update(
            phase=phase,
            message="Preparing color-word basis experiment",
            word_count=len(preview_words),
            word_source=word_source,
        )
        state = _ensure_checkpoint_state(
            output_dir=output_dir,
            name="color_word_basis",
            config={
                "grid_stride": grid_stride,
                "layers": None if layers is None else list(layers),
                "limit": resolved_limit,
                "logit_lens_top_token_count": logit_lens_top_token_count,
                "max_length": max_length,
                "max_new_tokens": max_new_tokens,
                "min_consensus_votes": min_consensus_votes,
                "model_name": model_name,
                "sae_activation_batch_size": sae_activation_batch_size,
                "sae_dictionary_multiplier": sae_dictionary_multiplier,
                "sae_dictionary_size": sae_dictionary_size,
                "sae_epochs": sae_epochs,
                "sae_l1_coefficient": sae_l1_coefficient,
                "sae_layer": sae_layer,
                "sae_learning_rate": sae_learning_rate,
                "sae_max_length": sae_max_length,
                "sae_prompt_format": sae_prompt_format,
                "sae_top_k": sae_top_k,
                "sae_train_batch_size": sae_train_batch_size,
                "sae_validation_fraction": sae_validation_fraction,
                "word_hash": _hash_words(preview_words),
                "word_list_path": None if resolved_word_list_path is None else str(resolved_word_list_path),
                "word_source": word_source,
            },
            resume=resume,
        )

        run_dir = output_dir / "run"
        logit_lens_dir = output_dir / "logit_lens"
        sae_train_dir = output_dir / "sae_train_word"
        sae_analysis_dir = output_dir / "sae_analysis_word"

        phase = "run"
        heartbeat.update(phase=phase, message="Running cross-format color-word anchors")
        run_summary = run_color_format_latent_experiment(
            output_dir=run_dir,
            model_name=model_name,
            word_list_path=resolved_word_list_path,
            word_preset=word_preset,
            limit=resolved_limit,
            formats=DEFAULT_FORMATS,
            layers=layers,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            grid_stride=grid_stride,
            min_consensus_votes=min_consensus_votes,
            resume=resume,
            device=device,
        )
        state["completed_run"] = True
        _save_checkpoint_state(output_dir, "color_word_basis", state)

        phase = "logit-lens"
        heartbeat.update(phase=phase, message="Running color-word logit lens")
        logit_lens_summary = run_color_logit_lens_experiment(
            output_dir=logit_lens_dir,
            model_name=model_name,
            word_list_path=resolved_word_list_path,
            word_preset=word_preset,
            limit=resolved_limit,
            formats=DEFAULT_FORMATS,
            layers=layers,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            top_token_count=logit_lens_top_token_count,
            resume=resume,
            device=device,
        )
        state["completed_logit_lens"] = True
        _save_checkpoint_state(output_dir, "color_word_basis", state)

        phase = "sae-train"
        heartbeat.update(phase=phase, message="Training SAE on color-word anchors")
        sae_training_summary = run_color_sae_training(
            output_dir=sae_train_dir,
            model_name=model_name,
            layer=sae_layer,
            prompt_format=sae_prompt_format,
            word_list_path=resolved_word_list_path,
            word_preset=word_preset,
            limit=resolved_limit,
            max_length=sae_max_length,
            activation_batch_size=sae_activation_batch_size,
            train_batch_size=sae_train_batch_size,
            device=device,
            dictionary_multiplier=sae_dictionary_multiplier,
            dictionary_size=sae_dictionary_size,
            top_k=sae_top_k,
            epochs=sae_epochs,
            learning_rate=sae_learning_rate,
            l1_coefficient=sae_l1_coefficient,
            validation_fraction=sae_validation_fraction,
            seed=sae_seed,
            resume=resume,
        )
        state["completed_sae_training"] = True
        _save_checkpoint_state(output_dir, "color_word_basis", state)

        phase = "sae-analyze"
        heartbeat.update(phase=phase, message="Analyzing SAE features on color-word run")
        sae_analysis_summary = run_color_sae_feature_analysis(
            sae_checkpoint_path=sae_train_dir / "sae_checkpoint.pt",
            color_run_dir=run_dir,
            output_dir=sae_analysis_dir,
            layer=sae_layer,
            format_name="word",
            batch_size=sae_train_batch_size,
            device="cpu" if device == "auto" else device,
        )
        state["completed_sae_analysis"] = True
        _save_checkpoint_state(output_dir, "color_word_basis", state)

        summary = {
            "generated_at_utc": _utc_now(),
            "logit_lens_dir": str(logit_lens_dir),
            "logit_lens_summary": logit_lens_summary,
            "model_name": model_name,
            "resume": resume,
            "run_dir": str(run_dir),
            "run_summary": run_summary,
            "sae_analysis_dir": str(sae_analysis_dir),
            "sae_analysis_summary": sae_analysis_summary,
            "sae_train_dir": str(sae_train_dir),
            "sae_training_summary": sae_training_summary,
            "word_count": len(preview_words),
            "word_list_path": None if resolved_word_list_path is None else str(resolved_word_list_path),
            "word_source": word_source,
        }
        _write_json(output_dir / "summary.json", summary)
        _write_color_word_basis_report(
            output_dir / "report.md",
            model_name=model_name,
            word_count=len(preview_words),
            word_source=word_source,
            run_summary=run_summary,
            logit_lens_summary=logit_lens_summary,
            sae_training_summary=sae_training_summary,
            sae_analysis_summary=sae_analysis_summary,
        )
        _write_color_word_basis_final_results(output_dir, summary=summary)
        phase = "completed"
        heartbeat.update(
            phase=phase,
            message="Color-word basis experiment complete",
            state="completed",
            best_cross_layer=run_summary.get("best_cross_layer"),
            sae_layer=sae_layer,
            word_count=len(preview_words),
        )
        return summary
    except Exception as error:
        heartbeat.fail(phase=phase, error=error)
        raise


def run_color_format_patch(
    *,
    run_dir: Path,
    output_dir: Path,
    model_name: str,
    source_format: str,
    target_format: str,
    layer: int,
    pairs_path: Path | None = None,
    limit: int | None = None,
    patch_mode: str = "replace",
    max_length: int = 96,
    max_new_tokens: int = 12,
    batch_size: int = 8,
    resume: bool = False,
    device: str = "auto",
) -> dict[str, Any]:
    if layer <= 0:
        raise ValueError("Residual patching currently supports layers >= 1.")
    if patch_mode not in {"replace", "add"}:
        raise ValueError("patch_mode must be one of: replace, add")
    np, torch, _PCA, _LogisticRegression, _KFold = _require_stack()
    output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = HeartbeatRecorder(output_dir, label="format-latent-patch")
    heartbeat.write_manifest(
        batch_size=batch_size,
        command="patch",
        device=device,
        layer=layer,
        limit=limit,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        model_name=model_name,
        pairs_path=None if pairs_path is None else str(pairs_path),
        patch_mode=patch_mode,
        resume=resume,
        run_dir=str(run_dir),
        source_format=source_format,
        target_format=target_format,
    )
    phase = "setup"
    try:
        if source_format not in FORMAT_PROMPTS or target_format not in FORMAT_PROMPTS:
            raise ValueError("Unsupported source or target format.")
        source_rows = _read_prediction_rows(run_dir / f"predictions_{source_format}.jsonl")
        target_rows = _read_prediction_rows(run_dir / f"predictions_{target_format}.jsonl")
        pairs = _load_pairs(
            pairs_path=pairs_path,
            source_rows=source_rows,
            target_rows=target_rows,
            limit=limit,
        )
        state = _ensure_checkpoint_state(
            output_dir=output_dir,
            name="patch",
            config={
                "batch_size": batch_size,
                "layer": layer,
                "limit": limit,
                "max_length": max_length,
                "max_new_tokens": max_new_tokens,
                "model_name": model_name,
                "pair_count": len(pairs),
                "patch_mode": patch_mode,
                "run_dir": str(run_dir),
                "source_format": source_format,
                "target_format": target_format,
            },
            resume=resume,
        )
        source_index = {row["word"]: index for index, row in enumerate(source_rows)}
        source_hidden = np.load(run_dir / "hidden_states" / source_format / f"layer_{layer:02d}.npy").astype(np.float32)

        tokenizer, model = create_generation_components(model_name)
        device_obj = _resolve_device(torch, device)
        model = model.to(device_obj)
        model.eval()
        blocks = _find_transformer_blocks(model)
        if layer - 1 >= len(blocks):
            raise ValueError(f"Requested patch layer {layer} but model only exposes {len(blocks)} blocks.")

        patch_rows: list[dict[str, Any]] = []
        total_batches = math.ceil(len(pairs) / batch_size)
        for batch_index, start in enumerate(range(0, len(pairs), batch_size), start=1):
            phase = "patch"
            cached_rows = _load_patch_batch_checkpoint(output_dir, batch_index) if resume else None
            if cached_rows is not None:
                patch_rows.extend(cached_rows)
                heartbeat.update(
                    phase=phase,
                    message=f"Loaded patch batch {batch_index}/{total_batches} from checkpoint",
                    batch_index=batch_index,
                    total_batches=total_batches,
                    processed_pairs=min(start + batch_size, len(pairs)),
                    total_pairs=len(pairs),
                )
                continue
            batch_pairs = pairs[start : start + batch_size]
            target_words = [pair["target_word"] for pair in batch_pairs]
            prompts = [
                _render_prompt(tokenizer, FORMAT_PROMPTS[target_format].format(word=word))
                for word in target_words
            ]
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded_device = _move_batch_to_device(encoded, device_obj)
            source_vectors = np.stack(
                [source_hidden[source_index[pair["source_word"]]] for pair in batch_pairs]
            )
            patch_vectors = torch.tensor(source_vectors, device=device_obj)
            last_positions = _non_padding_last_positions(encoded_device["attention_mask"])
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
                baseline_generated = model.generate(**generation_kwargs)
            prompt_length = int(encoded_device["input_ids"].shape[1])
            baseline_completions = tokenizer.batch_decode(
                baseline_generated[:, prompt_length:].detach().cpu(),
                skip_special_tokens=True,
            )

            hook_state = {"applied": False}

            def patch_hook(_module: Any, _args: Any, output: Any) -> Any:
                hidden, remainder = _coerce_hidden_output(output)
                if hook_state["applied"] or getattr(hidden, "shape", None) is None:
                    return output
                if hidden.shape[1] <= 1:
                    return output
                patched = hidden.clone()
                for row_index, last_position in enumerate(last_positions):
                    replacement = patch_vectors[row_index].to(dtype=patched.dtype)
                    if patch_mode == "replace":
                        patched[row_index, last_position, :] = replacement
                    else:
                        patched[row_index, last_position, :] = patched[row_index, last_position, :] + replacement
                hook_state["applied"] = True
                if remainder:
                    return (patched, *remainder)
                return patched

            handle = blocks[layer - 1].register_forward_hook(patch_hook)
            try:
                with torch.no_grad():
                    patched_generated = model.generate(**generation_kwargs)
            finally:
                handle.remove()

            patched_completions = tokenizer.batch_decode(
                patched_generated[:, prompt_length:].detach().cpu(),
                skip_special_tokens=True,
            )
            batch_rows: list[dict[str, Any]] = []
            for pair, baseline_raw, patched_raw in zip(
                batch_pairs,
                baseline_completions,
                patched_completions,
                strict=True,
            ):
                source_row = source_rows[source_index[pair["source_word"]]]
                baseline_parsed = parse_format_completion(target_format, baseline_raw)
                patched_parsed = parse_format_completion(target_format, patched_raw)
                source_family = source_row.get("color_family")
                baseline_family = baseline_parsed.color_family
                patched_family = patched_parsed.color_family
                batch_rows.append(
                    {
                        "baseline_matches_source": baseline_family == source_family,
                        "baseline_raw_completion": baseline_raw.strip(),
                        "baseline_target_family": baseline_family,
                        "changed": baseline_family != patched_family,
                        "moved_toward_source": patched_family == source_family and baseline_family != source_family,
                        "patch_mode": patch_mode,
                        "patched_matches_source": patched_family == source_family,
                        "patched_raw_completion": patched_raw.strip(),
                        "patched_target_family": patched_family,
                        "source_family": source_family,
                        "source_format": source_format,
                        "source_normalized_output": source_row.get("normalized_output"),
                        "source_word": pair["source_word"],
                        "target_format": target_format,
                        "target_word": pair["target_word"],
                        "layer": layer,
                    }
                )
            _save_patch_batch_checkpoint(output_dir, batch_index, batch_rows)
            patch_rows.extend(batch_rows)
            completed_batches = set(int(value) for value in state.get("completed_patch_batches", []))
            completed_batches.add(batch_index)
            state["completed_patch_batches"] = sorted(completed_batches)
            _save_checkpoint_state(output_dir, "patch", state)
            heartbeat.update(
                phase=phase,
                message=f"Patched batch {batch_index}/{total_batches}",
                batch_index=batch_index,
                total_batches=total_batches,
                processed_pairs=min(start + len(batch_pairs), len(pairs)),
                total_pairs=len(pairs),
            )

        _write_jsonl(output_dir / "patched_predictions.jsonl", patch_rows)
        changed_rate = _mean_or_none([1.0 if row["changed"] else 0.0 for row in patch_rows])
        moved_toward_source_rate = _mean_or_none(
            [1.0 if row["moved_toward_source"] else 0.0 for row in patch_rows]
        )
        patched_match_rate = _mean_or_none(
            [1.0 if row["patched_matches_source"] else 0.0 for row in patch_rows]
        )
        baseline_match_rate = _mean_or_none(
            [1.0 if row["baseline_matches_source"] else 0.0 for row in patch_rows]
        )
        summary = {
            "baseline_match_rate": baseline_match_rate,
            "changed_rate": changed_rate,
            "layer": layer,
            "moved_toward_source_rate": moved_toward_source_rate,
            "pair_count": len(patch_rows),
            "patch_mode": patch_mode,
            "patched_match_rate": patched_match_rate,
            "resume": resume,
            "source_format": source_format,
            "target_format": target_format,
        }
        _write_json(output_dir / "summary.json", summary)
        _write_patch_report(
            output_dir / "report.md",
            layer=layer,
            source_format=source_format,
            target_format=target_format,
            pair_count=len(patch_rows),
            changed_rate=changed_rate,
            moved_toward_source_rate=moved_toward_source_rate,
            patched_match_rate=patched_match_rate,
        )
        _write_patch_final_results(output_dir, summary=summary)
        phase = "completed"
        heartbeat.update(
            phase=phase,
            message="Residual patch run complete",
            state="completed",
            changed_rate=changed_rate,
            moved_toward_source_rate=moved_toward_source_rate,
            patched_match_rate=patched_match_rate,
        )
        return summary
    except Exception as error:
        heartbeat.fail(phase=phase, error=error)
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Color representation experiments for language models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="[legacy] Collect hidden states across color output formats.",
    )
    run_parser.add_argument("--model-name", required=True)
    run_parser.add_argument("--output-dir", required=True, type=Path)
    run_parser.add_argument("--word-list-path", type=Path)
    run_parser.add_argument("--word-preset", default="default", choices=WORD_PRESET_NAMES)
    run_parser.add_argument("--limit", type=int, default=1000)
    run_parser.add_argument("--formats", default="word,hex,rgb")
    run_parser.add_argument("--layers")
    run_parser.add_argument("--max-length", type=int, default=96)
    run_parser.add_argument("--max-new-tokens", type=int, default=12)
    run_parser.add_argument("--batch-size", type=int, default=16)
    run_parser.add_argument("--grid-stride", type=int, default=4)
    run_parser.add_argument("--min-consensus-votes", type=int, default=2)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--device", default="auto")

    patch_parser = subparsers.add_parser(
        "patch",
        help="[legacy] Patch source-format residuals into a target-format run.",
    )
    patch_parser.add_argument("--model-name", required=True)
    patch_parser.add_argument("--run-dir", required=True, type=Path)
    patch_parser.add_argument("--output-dir", required=True, type=Path)
    patch_parser.add_argument("--source-format", required=True, choices=sorted(FORMAT_PROMPTS))
    patch_parser.add_argument("--target-format", required=True, choices=sorted(FORMAT_PROMPTS))
    patch_parser.add_argument("--layer", required=True, type=int)
    patch_parser.add_argument("--pairs-path", type=Path)
    patch_parser.add_argument("--limit", type=int)
    patch_parser.add_argument("--patch-mode", default="replace", choices=("replace", "add"))
    patch_parser.add_argument("--max-length", type=int, default=96)
    patch_parser.add_argument("--max-new-tokens", type=int, default=12)
    patch_parser.add_argument("--batch-size", type=int, default=8)
    patch_parser.add_argument("--resume", action="store_true")
    patch_parser.add_argument("--device", default="auto")

    export_parser = subparsers.add_parser("export", help="Copy the key artifacts into a compact export bundle.")
    export_parser.add_argument("--run-dir", required=True, type=Path)
    export_parser.add_argument("--output-dir", required=True, type=Path)
    export_parser.add_argument("--patch-dir", type=Path)
    export_parser.add_argument("--logit-lens-dir", type=Path)

    basis_parser = subparsers.add_parser(
        "color-word-basis",
        help="[legacy] Run the older orchestrated color-word anchor study.",
    )
    basis_parser.add_argument("--model-name", required=True)
    basis_parser.add_argument("--output-dir", required=True, type=Path)
    basis_parser.add_argument("--word-list-path", type=Path)
    basis_parser.add_argument("--limit", type=int)
    basis_parser.add_argument("--layers")
    basis_parser.add_argument("--max-length", type=int, default=96)
    basis_parser.add_argument("--max-new-tokens", type=int, default=12)
    basis_parser.add_argument("--batch-size", type=int, default=16)
    basis_parser.add_argument("--grid-stride", type=int, default=4)
    basis_parser.add_argument("--min-consensus-votes", type=int, default=2)
    basis_parser.add_argument("--logit-lens-top-token-count", type=int, default=8)
    basis_parser.add_argument("--sae-layer", type=int, default=16)
    basis_parser.add_argument("--sae-prompt-format", default="word", choices=("word", "hex", "rgb"))
    basis_parser.add_argument("--sae-max-length", type=int, default=256)
    basis_parser.add_argument("--sae-activation-batch-size", type=int, default=32)
    basis_parser.add_argument("--sae-train-batch-size", type=int, default=256)
    basis_parser.add_argument("--sae-dictionary-multiplier", type=int, default=8)
    basis_parser.add_argument("--sae-dictionary-size", type=int)
    basis_parser.add_argument("--sae-top-k", type=int, default=32)
    basis_parser.add_argument("--sae-epochs", type=int, default=8)
    basis_parser.add_argument("--sae-learning-rate", type=float, default=1e-3)
    basis_parser.add_argument("--sae-l1-coefficient", type=float, default=1e-4)
    basis_parser.add_argument("--sae-validation-fraction", type=float, default=0.1)
    basis_parser.add_argument("--sae-seed", type=int, default=17)
    basis_parser.add_argument("--resume", action="store_true")
    basis_parser.add_argument("--device", default="auto")

    sae_train_parser = subparsers.add_parser(
        "sae-train",
        help="Train a custom SAE on one layer of color activations.",
    )
    sae_train_parser.add_argument("--model-name", required=True)
    sae_train_parser.add_argument("--output-dir", required=True, type=Path)
    sae_train_parser.add_argument("--layer", required=True, type=int)
    sae_train_parser.add_argument("--prompt-format", default="hex", choices=("word", "hex", "rgb"))
    sae_train_parser.add_argument("--word-list-path", type=Path)
    sae_train_parser.add_argument("--word-preset", default="default", choices=WORD_PRESET_NAMES)
    sae_train_parser.add_argument("--limit", type=int, default=10000)
    sae_train_parser.add_argument("--max-length", type=int, default=256)
    sae_train_parser.add_argument("--activation-batch-size", type=int, default=32)
    sae_train_parser.add_argument("--train-batch-size", type=int, default=256)
    sae_train_parser.add_argument("--dictionary-multiplier", type=int, default=8)
    sae_train_parser.add_argument("--dictionary-size", type=int)
    sae_train_parser.add_argument("--top-k", type=int, default=32)
    sae_train_parser.add_argument("--epochs", type=int, default=8)
    sae_train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    sae_train_parser.add_argument("--l1-coefficient", type=float, default=1e-4)
    sae_train_parser.add_argument("--validation-fraction", type=float, default=0.1)
    sae_train_parser.add_argument("--seed", type=int, default=17)
    sae_train_parser.add_argument("--resume", action="store_true")
    sae_train_parser.add_argument("--device", default="auto")

    sae_analyze_parser = subparsers.add_parser(
        "sae-analyze",
        help="Analyze a trained SAE against a completed color run.",
    )
    sae_analyze_parser.add_argument("--sae-checkpoint-path", required=True, type=Path)
    sae_analyze_parser.add_argument("--color-run-dir", required=True, type=Path)
    sae_analyze_parser.add_argument("--output-dir", required=True, type=Path)
    sae_analyze_parser.add_argument("--layer", required=True, type=int)
    sae_analyze_parser.add_argument("--format-name", default="all", choices=("word", "hex", "rgb", "all"))
    sae_analyze_parser.add_argument("--batch-size", type=int, default=256)
    sae_analyze_parser.add_argument("--device", default="cpu")

    sae_geometry_parser = subparsers.add_parser(
        "sae-geometry",
        help="Run an off-the-shelf SAE layer sweep on controlled `Color: X` prompts.",
    )
    sae_geometry_parser.add_argument("--model-name", required=True)
    sae_geometry_parser.add_argument("--output-dir", required=True, type=Path)
    sae_geometry_parser.add_argument("--sae-repo-id-or-path", default="andyrdt/saes-qwen2.5-7b-instruct")
    sae_geometry_parser.add_argument("--sae-layers", default="3,7,11,15,19,23,27")
    sae_geometry_parser.add_argument("--trainer-index", type=int, default=0)
    sae_geometry_parser.add_argument("--cache-dir", type=Path)
    sae_geometry_parser.add_argument("--word-list-path", type=Path)
    sae_geometry_parser.add_argument("--prompt-template", default="Color: {value}")
    sae_geometry_parser.add_argument("--word-limit", type=int)
    sae_geometry_parser.add_argument("--batch-size", type=int, default=64)
    sae_geometry_parser.add_argument("--encode-batch-size", type=int, default=256)
    sae_geometry_parser.add_argument("--max-length", type=int, default=64)
    sae_geometry_parser.add_argument("--no-word-catalog", action="store_true")
    sae_geometry_parser.add_argument("--no-anchor-word", action="store_true")
    sae_geometry_parser.add_argument("--no-anchor-hex", action="store_true")
    sae_geometry_parser.add_argument("--no-anchor-rgb", action="store_true")
    sae_geometry_parser.add_argument(
        "--catalog-formats",
        default="word",
        help="Comma-separated schemas to mirror the catalog into, e.g. `word,hex,rgb`.",
    )
    sae_geometry_parser.add_argument("--skip-silhouette", action="store_true")
    sae_geometry_parser.add_argument("--resume", action="store_true")
    sae_geometry_parser.add_argument("--device", default="auto")

    sae_intervene_parser = subparsers.add_parser(
        "sae-intervene",
        help="Inject a discovered SAE family direction into a prompt and measure the completion shift.",
    )
    sae_intervene_parser.add_argument("--model-name", required=True)
    sae_intervene_parser.add_argument("--geometry-dir", required=True, type=Path)
    sae_intervene_parser.add_argument("--output-dir", required=True, type=Path)
    sae_intervene_parser.add_argument("--layer", required=True, type=int)
    sae_intervene_parser.add_argument("--family", required=True)
    sae_intervene_parser.add_argument("--alpha-values", default="-8,-4,-2,-1,0,1,2,4,8")
    sae_intervene_parser.add_argument("--prompt-mode", default="blank_hex", choices=("blank_hex", "semantic_hex"))
    sae_intervene_parser.add_argument("--prompt-file", type=Path)
    sae_intervene_parser.add_argument("--output-format", choices=("hex", "word", "rgb", "description"))
    sae_intervene_parser.add_argument("--batch-size", type=int, default=8)
    sae_intervene_parser.add_argument("--max-length", type=int, default=128)
    sae_intervene_parser.add_argument("--max-new-tokens", type=int, default=16)
    sae_intervene_parser.add_argument("--resume", action="store_true")
    sae_intervene_parser.add_argument("--device", default="auto")

    logit_lens_parser = subparsers.add_parser(
        "logit-lens",
        help="[legacy] Project intermediate states through the unembedding and score color and format tokens.",
    )
    logit_lens_parser.add_argument("--model-name", required=True)
    logit_lens_parser.add_argument("--output-dir", required=True, type=Path)
    logit_lens_parser.add_argument("--word-list-path", type=Path)
    logit_lens_parser.add_argument("--word-preset", default="default", choices=WORD_PRESET_NAMES)
    logit_lens_parser.add_argument("--limit", type=int, default=1000)
    logit_lens_parser.add_argument("--formats", default="word,hex,rgb")
    logit_lens_parser.add_argument("--layers")
    logit_lens_parser.add_argument("--max-length", type=int, default=96)
    logit_lens_parser.add_argument("--max-new-tokens", type=int, default=12)
    logit_lens_parser.add_argument("--batch-size", type=int, default=16)
    logit_lens_parser.add_argument("--top-token-count", type=int, default=5)
    logit_lens_parser.add_argument("--resume", action="store_true")
    logit_lens_parser.add_argument("--device", default="auto")

    summarize_logit_lens_parser = subparsers.add_parser(
        "summarize-logit-lens",
        help="[legacy] Interpret an existing logit lens run without rerunning the model.",
    )
    summarize_logit_lens_parser.add_argument("--run-dir", required=True, type=Path)

    probe_compare_parser = subparsers.add_parser(
        "probe-compare",
        help="Compare linear probes on residual activations versus SAE codes from a saved geometry run.",
    )
    probe_compare_parser.add_argument("--geometry-dir", required=True, type=Path)
    probe_compare_parser.add_argument("--output-dir", required=True, type=Path)
    probe_compare_parser.add_argument("--layers")
    probe_compare_parser.add_argument("--label-mode", default="family", choices=("family", "color_word"))
    probe_compare_parser.add_argument("--schema-filter", help="Comma-separated schema filter, e.g. `word` or `word,hex,rgb`.")
    probe_compare_parser.add_argument("--center-mode", default="schema", choices=("none", "global", "schema"))
    probe_compare_parser.add_argument("--no-anchors", action="store_true")
    probe_compare_parser.add_argument("--no-catalog", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        formats = tuple(part.strip() for part in args.formats.split(",") if part.strip())
        run_color_format_latent_experiment(
            output_dir=args.output_dir,
            model_name=args.model_name,
            word_list_path=args.word_list_path,
            word_preset=args.word_preset,
            limit=args.limit,
            formats=formats,
            layers=_parse_layers(args.layers),
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            grid_stride=args.grid_stride,
            min_consensus_votes=args.min_consensus_votes,
            resume=args.resume,
            device=args.device,
        )
        return 0
    if args.command == "patch":
        run_color_format_patch(
            run_dir=args.run_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            source_format=args.source_format,
            target_format=args.target_format,
            layer=args.layer,
            pairs_path=args.pairs_path,
            limit=args.limit,
            patch_mode=args.patch_mode,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            resume=args.resume,
            device=args.device,
        )
        return 0
    if args.command == "export":
        export_final_results(
            run_dir=args.run_dir,
            output_dir=args.output_dir,
            patch_dir=args.patch_dir,
            logit_lens_dir=args.logit_lens_dir,
        )
        return 0
    if args.command == "color-word-basis":
        run_color_word_basis_experiment(
            output_dir=args.output_dir,
            model_name=args.model_name,
            word_list_path=args.word_list_path,
            limit=args.limit,
            layers=_parse_layers(args.layers),
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            grid_stride=args.grid_stride,
            min_consensus_votes=args.min_consensus_votes,
            logit_lens_top_token_count=args.logit_lens_top_token_count,
            sae_layer=args.sae_layer,
            sae_prompt_format=args.sae_prompt_format,
            sae_max_length=args.sae_max_length,
            sae_activation_batch_size=args.sae_activation_batch_size,
            sae_train_batch_size=args.sae_train_batch_size,
            sae_dictionary_multiplier=args.sae_dictionary_multiplier,
            sae_dictionary_size=args.sae_dictionary_size,
            sae_top_k=args.sae_top_k,
            sae_epochs=args.sae_epochs,
            sae_learning_rate=args.sae_learning_rate,
            sae_l1_coefficient=args.sae_l1_coefficient,
            sae_validation_fraction=args.sae_validation_fraction,
            sae_seed=args.sae_seed,
            resume=args.resume,
            device=args.device,
        )
        return 0
    if args.command == "sae-train":
        from .custom_sae import run_color_sae_training

        run_color_sae_training(
            output_dir=args.output_dir,
            model_name=args.model_name,
            layer=args.layer,
            prompt_format=args.prompt_format,
            word_list_path=args.word_list_path,
            word_preset=args.word_preset,
            limit=args.limit,
            max_length=args.max_length,
            activation_batch_size=args.activation_batch_size,
            train_batch_size=args.train_batch_size,
            device=args.device,
            dictionary_multiplier=args.dictionary_multiplier,
            dictionary_size=args.dictionary_size,
            top_k=args.top_k,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l1_coefficient=args.l1_coefficient,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            resume=args.resume,
        )
        return 0
    if args.command == "sae-analyze":
        from .custom_sae import run_color_sae_feature_analysis

        run_color_sae_feature_analysis(
            sae_checkpoint_path=args.sae_checkpoint_path,
            color_run_dir=args.color_run_dir,
            output_dir=args.output_dir,
            layer=args.layer,
            format_name=args.format_name,
            batch_size=args.batch_size,
            device=args.device,
        )
        return 0
    if args.command == "sae-geometry":
        from .sae_geometry import run_color_sae_geometry_experiment

        run_color_sae_geometry_experiment(
            output_dir=args.output_dir,
            model_name=args.model_name,
            sae_repo_id_or_path=args.sae_repo_id_or_path,
            sae_layers=_parse_layers(args.sae_layers),
            trainer_index=args.trainer_index,
            cache_dir=args.cache_dir,
            word_list_path=args.word_list_path,
            prompt_template=args.prompt_template,
            include_word_catalog=not args.no_word_catalog,
            include_anchor_word=not args.no_anchor_word,
            include_anchor_hex=not args.no_anchor_hex,
            include_anchor_rgb=not args.no_anchor_rgb,
            catalog_formats=tuple(part.strip() for part in args.catalog_formats.split(",") if part.strip()),
            word_limit=args.word_limit,
            batch_size=args.batch_size,
            max_length=args.max_length,
            encode_batch_size=args.encode_batch_size,
            compute_silhouette=not args.skip_silhouette,
            resume=args.resume,
            device=args.device,
        )
        return 0
    if args.command == "probe-compare":
        from .probe_compare import run_probe_comparison

        run_probe_comparison(
            geometry_dir=args.geometry_dir,
            output_dir=args.output_dir,
            layers=_parse_layers(args.layers),
            label_mode=args.label_mode,
            schema_filter=None
            if not args.schema_filter
            else tuple(part.strip() for part in args.schema_filter.split(",") if part.strip()),
            include_anchors=not args.no_anchors,
            include_catalog=not args.no_catalog,
            center_mode=args.center_mode,
        )
        return 0
    if args.command == "sae-intervene":
        from .sae_geometry import run_color_direction_intervention_experiment

        run_color_direction_intervention_experiment(
            output_dir=args.output_dir,
            geometry_dir=args.geometry_dir,
            model_name=args.model_name,
            layer=args.layer,
            family=args.family,
            alpha_values=args.alpha_values,
            prompt_mode=args.prompt_mode,
            prompt_file=args.prompt_file,
            output_format=args.output_format,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            resume=args.resume,
            device=args.device,
        )
        return 0
    if args.command == "logit-lens":
        from .logit_lens import run_color_logit_lens_experiment

        formats = tuple(part.strip() for part in args.formats.split(",") if part.strip())
        run_color_logit_lens_experiment(
            output_dir=args.output_dir,
            model_name=args.model_name,
            word_list_path=args.word_list_path,
            word_preset=args.word_preset,
            limit=args.limit,
            formats=formats,
            layers=_parse_layers(args.layers),
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            top_token_count=args.top_token_count,
            resume=args.resume,
            device=args.device,
        )
        return 0
    if args.command == "summarize-logit-lens":
        from .logit_lens import summarize_logit_lens_run

        summarize_logit_lens_run(args.run_dir)
        return 0
    raise AssertionError(f"Unhandled command {args.command!r}")
