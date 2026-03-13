from __future__ import annotations

import argparse
import colorsys
import hashlib
import html
import json
import math
import os
import platform
import re
import shutil
import socket
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hf import create_generation_components
from .word_lists import (
    WORD_PRESET_NAMES,
    bundled_color_word_list_path,
    default_words,
    find_system_word_list,
    preset_words,
    read_word_file,
)

FORMAT_PROMPTS: dict[str, str] = {
    "word": "What color do you associate with the word {word}? Reply with one color word.",
    "hex": "What color do you associate with the word {word}? Reply with a hex code.",
    "rgb": "What color do you associate with the word {word}? Reply with an RGB triplet like 255,0,0.",
}
DEFAULT_FORMATS = ("word", "hex", "rgb")
HEX_CODE_RE = re.compile(r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})")
RGB_TRIPLET_RE = re.compile(r"(?<!\d)(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})(?!\d)")
WORD_TOKEN_RE = re.compile(r"[a-zA-Z]+")
COLOR_WORD_SYNONYMS: dict[str, str] = {
    "red": "red",
    "scarlet": "red",
    "crimson": "red",
    "orange": "orange",
    "amber": "orange",
    "yellow": "yellow",
    "gold": "yellow",
    "green": "green",
    "emerald": "green",
    "blue": "blue",
    "azure": "blue",
    "cyan": "cyan",
    "teal": "cyan",
    "purple": "purple",
    "violet": "purple",
    "lavender": "purple",
    "magenta": "magenta",
    "pink": "magenta",
    "brown": "brown",
    "black": "black",
    "white": "white",
    "gray": "gray",
    "grey": "gray",
}
FAMILY_PALETTE: dict[str, str] = {
    "red": "#c93a3a",
    "orange": "#d9741b",
    "yellow": "#c7a400",
    "green": "#31844a",
    "cyan": "#1a8f9f",
    "blue": "#2b63c9",
    "purple": "#7a46af",
    "magenta": "#ba4b93",
    "brown": "#7c5b3d",
    "black": "#222222",
    "white": "#f4f1e8",
    "gray": "#8f8f8f",
}
FORMAT_STROKES: dict[str, str] = {
    "word": "#1a1a1a",
    "hex": "#4a4a4a",
    "rgb": "#7a7a7a",
}


def _require_stack() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import numpy as np  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        from sklearn.decomposition import PCA  # type: ignore[import-not-found]
        from sklearn.linear_model import LogisticRegression  # type: ignore[import-not-found]
        from sklearn.model_selection import KFold  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "The research stack is not installed. Use `pip install -e .` in the repo environment."
        ) from exc
    return np, torch, PCA, LogisticRegression, KFold


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True))
        handle.write("\n")


def _hash_words(words: list[str]) -> str:
    digest = hashlib.sha256()
    for word in words:
        digest.update(word.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _checkpoint_state_path(output_dir: Path, name: str) -> Path:
    return output_dir / "checkpoints" / f"{name}_state.json"


def _ensure_checkpoint_state(
    *,
    output_dir: Path,
    name: str,
    config: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    state_path = _checkpoint_state_path(output_dir, name)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and state_path.exists():
        state = _read_json(state_path)
        if state.get("config") != config:
            raise ValueError(f"Checkpoint config mismatch for {name}; start with a fresh output dir.")
        return state
    state = {
        "completed_formats": [],
        "completed_patch_batches": [],
        "config": config,
        "selected_layers": None,
        "updated_at_utc": _utc_now(),
    }
    _write_json(state_path, state)
    return state


def _save_checkpoint_state(output_dir: Path, name: str, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = _utc_now()
    _write_json(_checkpoint_state_path(output_dir, name), state)


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


class HeartbeatRecorder:
    def __init__(self, output_dir: Path, *, label: str) -> None:
        self.output_dir = output_dir
        self.label = label
        self.status_path = output_dir / "heartbeat_status.json"
        self.events_path = output_dir / "heartbeat_events.jsonl"
        self.start_time = time.monotonic()

    def write_manifest(self, **payload: Any) -> None:
        manifest = {
            "created_at_utc": _utc_now(),
            "cwd": str(Path.cwd()),
            "hostname": socket.gethostname(),
            "label": self.label,
            "pid": os.getpid(),
            "python_version": platform.python_version(),
        }
        manifest.update(payload)
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def update(self, *, phase: str, message: str, **payload: Any) -> None:
        row = {
            "event": "heartbeat",
            "label": self.label,
            "message": message,
            "phase": phase,
            "runtime_seconds": round(time.monotonic() - self.start_time, 3),
            "state": payload.get("state", "running"),
            "updated_at_utc": _utc_now(),
        }
        row.update(payload)
        self.status_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
        _append_jsonl_row(self.events_path, row)
        print(f"[{self.label}:{phase}] {message}", flush=True)

    def event(self, *, phase: str, message: str, **payload: Any) -> None:
        row = {
            "event": "info",
            "label": self.label,
            "message": message,
            "phase": phase,
            "runtime_seconds": round(time.monotonic() - self.start_time, 3),
            "updated_at_utc": _utc_now(),
        }
        row.update(payload)
        _append_jsonl_row(self.events_path, row)

    def fail(self, *, phase: str, error: BaseException) -> None:
        self.update(
            phase=phase,
            message=f"{type(error).__name__}: {error}",
            error_type=type(error).__name__,
            state="failed",
        )


@dataclass(frozen=True)
class ParsedCompletion:
    normalized_output: str | None
    color_family: str | None
    temperature: str | None


def _normalize_hex_code(raw_completion: str) -> str | None:
    match = HEX_CODE_RE.search(raw_completion)
    if match is None:
        return None
    value = match.group(0).lower()
    if len(value) == 4:
        return "#" + "".join(character * 2 for character in value[1:])
    return value


def _hex_to_rgb(hex_code: str) -> tuple[int, int, int]:
    return (
        int(hex_code[1:3], 16),
        int(hex_code[3:5], 16),
        int(hex_code[5:7], 16),
    )


def _color_family_from_rgb(red: int, green: int, blue: int) -> str:
    red_f = red / 255.0
    green_f = green / 255.0
    blue_f = blue / 255.0
    hue, saturation, value = colorsys.rgb_to_hsv(red_f, green_f, blue_f)
    hue_degrees = hue * 360.0

    if value <= 0.15:
        return "black"
    if saturation <= 0.12 and value >= 0.9:
        return "white"
    if saturation <= 0.15:
        return "gray"
    if 15.0 <= hue_degrees < 45.0 and value < 0.65:
        return "brown"
    if hue_degrees < 15.0 or hue_degrees >= 345.0:
        return "red"
    if hue_degrees < 45.0:
        return "orange"
    if hue_degrees < 75.0:
        return "yellow"
    if hue_degrees < 165.0:
        return "green"
    if hue_degrees < 195.0:
        return "cyan"
    if hue_degrees < 255.0:
        return "blue"
    if hue_degrees < 300.0:
        return "purple"
    return "magenta"


def _color_family_from_hex(hex_code: str) -> str:
    return _color_family_from_rgb(*_hex_to_rgb(hex_code))


def _temperature_from_family(family: str | None) -> str | None:
    if family is None:
        return None
    if family in {"red", "orange", "yellow", "brown", "magenta"}:
        return "warm"
    if family in {"green", "cyan", "blue", "purple"}:
        return "cool"
    return "neutral"


def parse_format_completion(format_name: str, raw_completion: str) -> ParsedCompletion:
    stripped = raw_completion.strip()
    if format_name == "word":
        for token in WORD_TOKEN_RE.findall(stripped.lower()):
            family = COLOR_WORD_SYNONYMS.get(token)
            if family is not None:
                return ParsedCompletion(
                    normalized_output=token,
                    color_family=family,
                    temperature=_temperature_from_family(family),
                )
        return ParsedCompletion(None, None, None)
    if format_name == "hex":
        hex_code = _normalize_hex_code(stripped)
        if hex_code is None:
            return ParsedCompletion(None, None, None)
        family = _color_family_from_hex(hex_code)
        return ParsedCompletion(
            normalized_output=hex_code,
            color_family=family,
            temperature=_temperature_from_family(family),
        )
    if format_name == "rgb":
        match = RGB_TRIPLET_RE.search(stripped)
        if match is None:
            return ParsedCompletion(None, None, None)
        values = tuple(int(group) for group in match.groups())
        if any(value < 0 or value > 255 for value in values):
            return ParsedCompletion(None, None, None)
        normalized = ",".join(str(value) for value in values)
        family = _color_family_from_rgb(*values)
        return ParsedCompletion(
            normalized_output=normalized,
            color_family=family,
            temperature=_temperature_from_family(family),
        )
    raise ValueError(f"Unsupported format {format_name!r}")


def _render_prompt(tokenizer: Any, prompt: str) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            return str(
                apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except Exception:
            return prompt
    return prompt


def _resolve_device(torch: Any, requested_device: str) -> Any:
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _non_padding_last_positions(attention_mask: Any) -> list[int]:
    positions: list[int] = []
    for row in attention_mask.detach().cpu().tolist():
        indices = [index for index, value in enumerate(row) if int(value) == 1]
        if not indices:
            raise RuntimeError("Encountered an empty prompt after tokenization.")
        positions.append(indices[-1])
    return positions


def _select_layers(hidden_states: tuple[Any, ...], requested_layers: tuple[int, ...] | None) -> tuple[int, ...]:
    if requested_layers is None:
        return tuple(range(len(hidden_states)))
    valid_layers: list[int] = []
    upper_bound = len(hidden_states) - 1
    for layer in requested_layers:
        if layer < 0 or layer > upper_bound:
            raise ValueError(f"Requested layer {layer} outside valid range 0..{upper_bound}")
        valid_layers.append(layer)
    return tuple(dict.fromkeys(valid_layers))


def _parse_layers(text: str | None) -> tuple[int, ...] | None:
    if text is None or not text.strip():
        return None
    values = []
    for chunk in text.split(","):
        value = chunk.strip()
        if not value:
            continue
        values.append(int(value))
    return tuple(values)


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


def _projection_bounds(coords: Any) -> tuple[float, float, float, float]:
    xs = coords[:, 0]
    ys = coords[:, 1]
    min_x = float(xs.min())
    max_x = float(xs.max())
    min_y = float(ys.min())
    max_y = float(ys.max())
    if math.isclose(min_x, max_x):
        min_x -= 1.0
        max_x += 1.0
    if math.isclose(min_y, max_y):
        min_y -= 1.0
        max_y += 1.0
    return min_x, max_x, min_y, max_y


def _format_marker(
    *,
    x: float,
    y: float,
    format_name: str,
    fill: str,
    stroke: str,
    title: str,
) -> str:
    escaped = html.escape(title)
    if format_name == "word":
        return (
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.4" fill="{fill}" stroke="{stroke}" stroke-width="1.1">'
            f"<title>{escaped}</title></circle>"
        )
    if format_name == "hex":
        return (
            f'<rect x="{x - 4.1:.2f}" y="{y - 4.1:.2f}" width="8.2" height="8.2" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.1"><title>{escaped}</title></rect>'
        )
    points = (
        f"{x:.2f},{y - 4.8:.2f} "
        f"{x + 4.8:.2f},{y:.2f} "
        f"{x:.2f},{y + 4.8:.2f} "
        f"{x - 4.8:.2f},{y:.2f}"
    )
    return (
        f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" stroke-width="1.1">'
        f"<title>{escaped}</title></polygon>"
    )


def _write_shared_pca_svg(
    path: Path,
    *,
    coords: Any,
    points: list[dict[str, Any]],
    title: str,
    subtitle: str,
    pc1_variance: float,
    pc2_variance: float,
) -> None:
    width = 980.0
    height = 760.0
    margin = 80.0
    inner_width = width - 2.0 * margin
    inner_height = height - 2.0 * margin
    min_x, max_x, min_y, max_y = _projection_bounds(coords)
    span_x = max_x - min_x
    span_y = max_y - min_y

    def project_x(value: float) -> float:
        return margin + ((value - min_x) / span_x) * inner_width

    def project_y(value: float) -> float:
        return height - margin - ((value - min_y) / span_y) * inner_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" viewBox="0 0 {int(width)} {int(height)}">',
        '<rect width="100%" height="100%" fill="#fbfaf7" />',
        f'<text x="{margin}" y="34" font-family="Helvetica, Arial, sans-serif" font-size="25" fill="#111111">{html.escape(title)}</text>',
        f'<text x="{margin}" y="58" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#444444">{html.escape(subtitle)}</text>',
        (
            f'<text x="{margin}" y="78" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#555555">'
            f"PC1 {pc1_variance:.1%}   PC2 {pc2_variance:.1%}   points={len(points)}</text>"
        ),
        f'<rect x="{margin}" y="{margin}" width="{inner_width}" height="{inner_height}" fill="none" stroke="#bbb5ab" stroke-width="1" />',
    ]
    for point, coord in zip(points, coords.tolist(), strict=True):
        fill = FAMILY_PALETTE.get(point["consensus_color_family"], "#777777")
        stroke = FORMAT_STROKES.get(point["format"], "#222222")
        title_text = (
            f'{point["word"]} | format={point["format"]} | consensus={point["consensus_color_family"]} '
            f'| parsed={point["source_color_family"]}'
        )
        lines.append(
            _format_marker(
                x=project_x(float(coord[0])),
                y=project_y(float(coord[1])),
                format_name=point["format"],
                fill=fill,
                stroke=stroke,
                title=title_text,
            )
        )

    legend_x = width - 235.0
    legend_y = 98.0
    lines.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="170" height="118" rx="8" fill="#ffffff" stroke="#d0cbc2" stroke-width="1" opacity="0.95" />'
    )
    for index, format_name in enumerate(DEFAULT_FORMATS):
        y = legend_y + 24.0 + index * 24.0
        lines.append(
            _format_marker(
                x=legend_x + 18.0,
                y=y,
                format_name=format_name,
                fill="#e5ddd2",
                stroke=FORMAT_STROKES[format_name],
                title=f"format={format_name}",
            )
        )
        lines.append(
            f'<text x="{legend_x + 32.0}" y="{y + 4.0}" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#333333">{html.escape(format_name)}</text>'
        )
    families = ["red", "orange", "yellow", "green", "blue", "purple", "magenta", "brown"]
    for index, family in enumerate(families):
        y = legend_y + 16.0 + index * 12.0
        x = legend_x + 92.0
        lines.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{FAMILY_PALETTE[family]}" stroke="#333333" stroke-width="0.6" />')
        lines.append(
            f'<text x="{x + 10.0}" y="{y + 4.0}" font-family="Helvetica, Arial, sans-serif" font-size="10" fill="#333333">{html.escape(family)}</text>'
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_shared_pca_grid_svg(path: Path, *, layer_specs: list[dict[str, Any]], stride: int) -> None:
    if not layer_specs:
        path.write_text("", encoding="utf-8")
        return
    selected_specs = layer_specs[:: max(1, stride)]
    if selected_specs[-1]["layer"] != layer_specs[-1]["layer"]:
        selected_specs.append(layer_specs[-1])
    columns = min(3, max(1, len(selected_specs)))
    rows = math.ceil(len(selected_specs) / columns)
    panel_width = 360.0
    panel_height = 290.0
    outer_margin = 28.0
    width = columns * panel_width + 2.0 * outer_margin
    height = rows * panel_height + 2.0 * outer_margin + 34.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" viewBox="0 0 {int(width)} {int(height)}">',
        '<rect width="100%" height="100%" fill="#fbfaf7" />',
        f'<text x="{outer_margin}" y="26" font-family="Helvetica, Arial, sans-serif" font-size="22" fill="#111111">Shared-PCA layer grid</text>',
    ]
    for index, spec in enumerate(selected_specs):
        col = index % columns
        row = index // columns
        panel_x = outer_margin + col * panel_width
        panel_y = outer_margin + 14.0 + row * panel_height
        inner_margin = 38.0
        inner_width = panel_width - 2.0 * inner_margin
        inner_height = panel_height - 2.0 * inner_margin - 18.0
        coords = spec["coords"]
        points = spec["points"]
        min_x, max_x, min_y, max_y = _projection_bounds(coords)
        span_x = max_x - min_x
        span_y = max_y - min_y

        def project_x(value: float) -> float:
            return panel_x + inner_margin + ((value - min_x) / span_x) * inner_width

        def project_y(value: float) -> float:
            return panel_y + panel_height - inner_margin - ((value - min_y) / span_y) * inner_height

        lines.extend(
            [
                f'<rect x="{panel_x}" y="{panel_y}" width="{panel_width - 10.0}" height="{panel_height - 10.0}" rx="8" fill="#ffffff" stroke="#d0cbc2" stroke-width="1" />',
                f'<text x="{panel_x + 16.0}" y="{panel_y + 24.0}" font-family="Helvetica, Arial, sans-serif" font-size="16" fill="#111111">Layer {spec["layer"]}</text>',
                (
                    f'<text x="{panel_x + 16.0}" y="{panel_y + 42.0}" font-family="Helvetica, Arial, sans-serif" '
                    f'font-size="11" fill="#555555">PC1 {spec["pc1_variance"]:.1%} | PC2 {spec["pc2_variance"]:.1%}</text>'
                ),
                f'<rect x="{panel_x + inner_margin}" y="{panel_y + inner_margin + 14.0}" width="{inner_width}" height="{inner_height}" fill="none" stroke="#bbb5ab" stroke-width="1" />',
            ]
        )
        for point, coord in zip(points, coords.tolist(), strict=True):
            lines.append(
                _format_marker(
                    x=project_x(float(coord[0])),
                    y=project_y(float(coord[1])),
                    format_name=point["format"],
                    fill=FAMILY_PALETTE.get(point["consensus_color_family"], "#777777"),
                    stroke=FORMAT_STROKES.get(point["format"], "#222222"),
                    title=(
                        f'{point["word"]} | format={point["format"]} | '
                        f'consensus={point["consensus_color_family"]}'
                    ),
                )
            )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_accuracy_curve_svg(path: Path, layer_rows: list[dict[str, Any]]) -> None:
    width = 920.0
    height = 520.0
    margin = 72.0
    inner_width = width - 2.0 * margin
    inner_height = height - 2.0 * margin
    rows = [
        row
        for row in layer_rows
        if row.get("within_schema_mean") is not None or row.get("cross_format_mean") is not None
    ]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    min_layer = float(min(row["layer"] for row in rows))
    max_layer = float(max(row["layer"] for row in rows))
    if math.isclose(min_layer, max_layer):
        max_layer = min_layer + 1.0

    def project_x(value: float) -> float:
        return margin + ((value - min_layer) / (max_layer - min_layer)) * inner_width

    def project_y(value: float) -> float:
        return height - margin - value * inner_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" viewBox="0 0 {int(width)} {int(height)}">',
        '<rect width="100%" height="100%" fill="#fbfaf7" />',
        f'<text x="{margin}" y="34" font-family="Helvetica, Arial, sans-serif" font-size="24" fill="#111111">Within-schema and cross-format probe accuracy</text>',
        f'<rect x="{margin}" y="{margin}" width="{inner_width}" height="{inner_height}" fill="none" stroke="#bbb5ab" stroke-width="1" />',
    ]
    for guide in [0.2, 0.4, 0.6, 0.8]:
        y = project_y(guide)
        lines.append(
            f'<line x1="{margin}" y1="{y:.2f}" x2="{width - margin}" y2="{y:.2f}" stroke="#e2ddd5" stroke-width="1" />'
        )
        lines.append(
            f'<text x="{margin - 36.0}" y="{y + 4.0:.2f}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#666666">{guide:.1f}</text>'
        )
    within_points = [
        (project_x(float(row["layer"])), project_y(float(row["within_schema_mean"])))
        for row in rows
        if row.get("within_schema_mean") is not None
    ]
    cross_points = [
        (project_x(float(row["layer"])), project_y(float(row["cross_format_mean"])))
        for row in rows
        if row.get("cross_format_mean") is not None
    ]
    if len(within_points) >= 2:
        path_data = " ".join(
            f"{'M' if index == 0 else 'L'} {x:.2f} {y:.2f}" for index, (x, y) in enumerate(within_points)
        )
        lines.append(f'<path d="{path_data}" fill="none" stroke="#1f6f5f" stroke-width="3" />')
    if len(cross_points) >= 2:
        path_data = " ".join(
            f"{'M' if index == 0 else 'L'} {x:.2f} {y:.2f}" for index, (x, y) in enumerate(cross_points)
        )
        lines.append(f'<path d="{path_data}" fill="none" stroke="#9b3d7a" stroke-width="3" />')
    for x, y in within_points:
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.6" fill="#1f6f5f" />')
    for x, y in cross_points:
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.6" fill="#9b3d7a" />')
    lines.extend(
        [
            f'<circle cx="{width - margin - 120.0}" cy="{margin - 26.0}" r="4" fill="#1f6f5f" />',
            f'<text x="{width - margin - 108.0}" y="{margin - 22.0}" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#333333">within-schema exact-output mean</text>',
            f'<circle cx="{width - margin - 120.0}" cy="{margin - 8.0}" r="4" fill="#9b3d7a" />',
            f'<text x="{width - margin - 108.0}" y="{margin - 4.0}" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#333333">cross-format family mean</text>',
        ]
    )
    for row in rows:
        x = project_x(float(row["layer"]))
        lines.append(
            f'<text x="{x - 5.0:.2f}" y="{height - margin + 20.0:.2f}" font-family="Helvetica, Arial, sans-serif" font-size="10" fill="#666666">{row["layer"]}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _mean_or_none(values: list[float]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def _fit_probe_accuracy(
    *,
    np: Any,
    LogisticRegression: Any,
    KFold: Any,
    source_features: Any,
    target_features: Any,
    labels: list[str],
) -> float | None:
    if len(labels) < 4:
        return None
    label_set = sorted(set(labels))
    if len(label_set) < 2:
        return None
    n_splits = min(5, len(labels))
    if n_splits < 2:
        return None
    label_to_index = {label: index for index, label in enumerate(label_set)}
    encoded_labels = np.array([label_to_index[label] for label in labels], dtype=np.int64)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores: list[float] = []
    for train_indices, test_indices in splitter.split(source_features):
        train_labels = encoded_labels[train_indices]
        if len(set(train_labels.tolist())) < 2:
            continue
        classifier = LogisticRegression(max_iter=2000)
        classifier.fit(source_features[train_indices], train_labels)
        predictions = classifier.predict(target_features[test_indices])
        scores.append(float((predictions == encoded_labels[test_indices]).mean()))
    return _mean_or_none(scores)


def _fit_transfer_accuracy(
    *,
    np: Any,
    LogisticRegression: Any,
    KFold: Any,
    source_matrix: Any,
    target_matrix: Any,
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    consensus_by_word: dict[str, str],
) -> float | None:
    source_index = {row["word"]: index for index, row in enumerate(source_rows)}
    target_index = {row["word"]: index for index, row in enumerate(target_rows)}
    common_words = [word for word in consensus_by_word if word in source_index and word in target_index]
    if len(common_words) < 4:
        return None
    source_features = np.stack(
        [source_matrix[source_index[word]].astype(np.float32) for word in common_words]
    )
    target_features = np.stack(
        [target_matrix[target_index[word]].astype(np.float32) for word in common_words]
    )
    labels = [consensus_by_word[word] for word in common_words]
    return _fit_probe_accuracy(
        np=np,
        LogisticRegression=LogisticRegression,
        KFold=KFold,
        source_features=source_features,
        target_features=target_features,
        labels=labels,
    )


def _fit_within_schema_accuracy(
    *,
    np: Any,
    LogisticRegression: Any,
    KFold: Any,
    matrix: Any,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    valid_indices = [
        index for index, row in enumerate(rows) if isinstance(row.get("normalized_output"), str)
    ]
    if len(valid_indices) < 4:
        return {"accuracy": None, "label_count": 0, "sample_count": len(valid_indices)}
    features = matrix[np.array(valid_indices, dtype=np.int64)].astype(np.float32)
    labels = [str(rows[index]["normalized_output"]) for index in valid_indices]
    accuracy = _fit_probe_accuracy(
        np=np,
        LogisticRegression=LogisticRegression,
        KFold=KFold,
        source_features=features,
        target_features=features,
        labels=labels,
    )
    return {
        "accuracy": accuracy,
        "label_count": len(set(labels)),
        "sample_count": len(labels),
    }


def _find_transformer_blocks(model: Any) -> Any:
    for candidate in (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
        getattr(getattr(model, "gpt_neox", None), "layers", None),
    ):
        if candidate is not None:
            return candidate
    raise RuntimeError("Could not find transformer blocks for residual patching on this model.")


def _coerce_hidden_output(output: Any) -> tuple[Any, tuple[Any, ...]]:
    if isinstance(output, tuple):
        return output[0], tuple(output[1:])
    return output, ()


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


def _build_layer_analysis(
    *,
    output_dir: Path,
    formats: tuple[str, ...],
    layers: tuple[int, ...],
    predictions_by_format: dict[str, list[dict[str, Any]]],
    consensus_by_word: dict[str, str],
    grid_stride: int,
    heartbeat: HeartbeatRecorder,
    np: Any,
    PCA: Any,
    LogisticRegression: Any,
    KFold: Any,
) -> dict[str, Any]:
    hidden_dir = output_dir / "hidden_states"
    points_rows: list[dict[str, Any]] = []
    cross_transfer_rows: list[dict[str, Any]] = []
    within_schema_rows: list[dict[str, Any]] = []
    grid_specs: list[dict[str, Any]] = []
    layer_summary_rows: list[dict[str, Any]] = []
    for layer_position, layer in enumerate(layers, start=1):
        heartbeat.update(
            phase="analyze",
            message=f"Analyzing layer {layer}",
            completed_layers=layer_position - 1,
            current_layer=layer,
            total_layers=len(layers),
        )
        layer_matrices = {
            format_name: np.load(hidden_dir / format_name / f"layer_{layer:02d}.npy").astype(np.float32)
            for format_name in formats
        }
        pooled_vectors = []
        pooled_points: list[dict[str, Any]] = []
        for format_name in formats:
            rows = predictions_by_format[format_name]
            matrix = layer_matrices[format_name]
            for row_index, row in enumerate(rows):
                consensus = consensus_by_word.get(row["word"])
                if consensus is None:
                    continue
                pooled_vectors.append(matrix[row_index])
                pooled_points.append(
                    {
                        "consensus_color_family": consensus,
                        "format": format_name,
                        "source_color_family": row.get("color_family"),
                        "word": row["word"],
                    }
                )
        if len(pooled_vectors) < 2:
            continue
        pooled_array = np.stack(pooled_vectors)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(pooled_array)
        pc1_variance = float(pca.explained_variance_ratio_[0]) if len(pca.explained_variance_ratio_) > 0 else 0.0
        pc2_variance = float(pca.explained_variance_ratio_[1]) if len(pca.explained_variance_ratio_) > 1 else 0.0
        spec = {
            "coords": coords,
            "layer": layer,
            "pc1_variance": pc1_variance,
            "pc2_variance": pc2_variance,
            "points": pooled_points,
        }
        grid_specs.append(spec)
        _write_shared_pca_svg(
            output_dir / f"layer_{layer:02d}_shared_pca.svg",
            coords=coords,
            points=pooled_points,
            title=f"Layer {layer} shared PCA",
            subtitle="Pooled residual states from word, hex, and rgb prompts, colored by consensus family.",
            pc1_variance=pc1_variance,
            pc2_variance=pc2_variance,
        )
        for point, coord in zip(pooled_points, coords.tolist(), strict=True):
            points_rows.append(
                {
                    "consensus_color_family": point["consensus_color_family"],
                    "format": point["format"],
                    "layer": layer,
                    "source_color_family": point["source_color_family"],
                    "word": point["word"],
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                }
            )

        within_scores: list[float] = []
        cross_scores: list[float] = []
        for format_name in formats:
            within_schema = _fit_within_schema_accuracy(
                np=np,
                LogisticRegression=LogisticRegression,
                KFold=KFold,
                matrix=layer_matrices[format_name],
                rows=predictions_by_format[format_name],
            )
            within_schema_rows.append(
                {
                    "accuracy": within_schema["accuracy"],
                    "format": format_name,
                    "label_count": within_schema["label_count"],
                    "label_mode": "schema_output",
                    "layer": layer,
                    "sample_count": within_schema["sample_count"],
                }
            )
            if within_schema["accuracy"] is not None:
                within_scores.append(float(within_schema["accuracy"]))
        for source_format in formats:
            for target_format in formats:
                if source_format == target_format:
                    continue
                accuracy = _fit_transfer_accuracy(
                    np=np,
                    LogisticRegression=LogisticRegression,
                    KFold=KFold,
                    source_matrix=layer_matrices[source_format],
                    target_matrix=layer_matrices[target_format],
                    source_rows=predictions_by_format[source_format],
                    target_rows=predictions_by_format[target_format],
                    consensus_by_word=consensus_by_word,
                )
                cross_transfer_rows.append(
                    {
                        "accuracy": accuracy,
                        "layer": layer,
                        "label_mode": "color_family_consensus",
                        "source_format": source_format,
                        "target_format": target_format,
                    }
                )
                if accuracy is None:
                    continue
                cross_scores.append(accuracy)
        layer_summary_rows.append(
            {
                "cross_format_mean": _mean_or_none(cross_scores),
                "layer": layer,
                "within_schema_mean": _mean_or_none(within_scores),
            }
        )

    _write_jsonl(output_dir / "shared_pca_points.jsonl", points_rows)
    _write_jsonl(output_dir / "cross_format_probe_transfer.jsonl", cross_transfer_rows)
    _write_jsonl(output_dir / "probe_transfer.jsonl", cross_transfer_rows)
    _write_jsonl(output_dir / "within_schema_probe_accuracy.jsonl", within_schema_rows)
    _write_jsonl(output_dir / "layer_summary.jsonl", layer_summary_rows)
    _write_shared_pca_grid_svg(output_dir / "shared_pca_grid.svg", layer_specs=grid_specs, stride=grid_stride)
    _write_accuracy_curve_svg(output_dir / "format_transfer_curve.svg", layer_summary_rows)

    best_cross_row = max(
        (row for row in layer_summary_rows if row.get("cross_format_mean") is not None),
        key=lambda row: float(row["cross_format_mean"]),
        default=None,
    )
    best_within_schema_rows: dict[str, dict[str, Any]] = {}
    for format_name in formats:
        best_row = max(
            (
                row
                for row in within_schema_rows
                if row["format"] == format_name and row.get("accuracy") is not None
            ),
            key=lambda row: float(row["accuracy"]),
            default=None,
        )
        if best_row is not None:
            best_within_schema_rows[format_name] = best_row
    return {
        "best_cross_layer": None if best_cross_row is None else best_cross_row["layer"],
        "best_cross_mean_accuracy": None
        if best_cross_row is None
        else float(best_cross_row["cross_format_mean"]),
        "best_within_schema_accuracy_by_format": {
            format_name: None
            if format_name not in best_within_schema_rows
            else float(best_within_schema_rows[format_name]["accuracy"])
            for format_name in formats
        },
        "best_within_schema_layer_by_format": {
            format_name: None
            if format_name not in best_within_schema_rows
            else int(best_within_schema_rows[format_name]["layer"])
            for format_name in formats
        },
        "cross_transfer_rows": cross_transfer_rows,
        "layer_summary_rows": layer_summary_rows,
        "within_schema_rows": within_schema_rows,
    }


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


def _read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
            word_limit=args.word_limit,
            batch_size=args.batch_size,
            max_length=args.max_length,
            encode_batch_size=args.encode_batch_size,
            compute_silhouette=not args.skip_silhouette,
            resume=args.resume,
            device=args.device,
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
