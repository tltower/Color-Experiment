from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_layers(text: str | None) -> tuple[int, ...] | None:
    if text is None or not text.strip():
        return None
    return tuple(int(chunk.strip()) for chunk in text.split(",") if chunk.strip())


def available_direction_layers(geometry_dir: Path) -> tuple[int, ...]:
    layers: list[int] = []
    for path in sorted(geometry_dir.glob("layer_*/directions")):
        layers.append(int(path.parent.name.split("_", 1)[1]))
    if not layers:
        raise FileNotFoundError(f"No layer_XX/directions directories found under {geometry_dir}")
    return tuple(layers)


def available_activation_layers(geometry_dir: Path) -> tuple[int, ...]:
    layers: list[int] = []
    for path in sorted((geometry_dir / "activations").glob("layer_*.npy")):
        layers.append(int(path.stem.split("_", 1)[1]))
    if not layers:
        raise FileNotFoundError(f"No activations/layer_XX.npy files found under {geometry_dir}")
    return tuple(layers)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def cosine_similarity_matrix(values: Any) -> Any:
    norms = (values**2).sum(axis=1, keepdims=True) ** 0.5 + 1e-8
    normalized = values / norms
    return normalized @ normalized.T


__all__ = [
    "available_activation_layers",
    "available_direction_layers",
    "cosine_similarity_matrix",
    "parse_layers",
    "read_json",
    "read_jsonl",
    "write_json",
    "write_jsonl",
]
