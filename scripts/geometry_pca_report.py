#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

CORE_COLOR_HEX: dict[str, str] = {
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
FORMAT_STROKES: dict[str, str] = {
    "word": "#1a1a1a",
    "hex": "#4a4a4a",
    "rgb": "#7a7a7a",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _available_layers(geometry_dir: Path) -> tuple[int, ...]:
    layers: list[int] = []
    for path in sorted(geometry_dir.glob("layer_*/encoded_features.npy")):
        suffix = path.parent.name.split("_", 1)[1]
        layers.append(int(suffix))
    if not layers:
        raise FileNotFoundError(f"No layer_XX/encoded_features.npy files found under {geometry_dir}")
    return tuple(layers)


def _parse_layers(value: str | None) -> tuple[int, ...] | None:
    if value is None or not value.strip():
        return None
    return tuple(int(chunk.strip()) for chunk in value.split(",") if chunk.strip())


def _normalize_hex(value: str) -> str:
    lowered = value.strip().lower()
    if lowered.startswith("#") and len(lowered) == 4:
        return "#" + "".join(character * 2 for character in lowered[1:])
    return lowered


def _rgb_to_hex(value: str) -> str:
    channels = [int(chunk.strip()) for chunk in value.split(",")]
    bounded = [max(0, min(255, channel)) for channel in channels[:3]]
    while len(bounded) < 3:
        bounded.append(0)
    return f"#{bounded[0]:02x}{bounded[1]:02x}{bounded[2]:02x}"


def _display_color(row: dict[str, Any]) -> str:
    schema = str(row.get("schema"))
    value = str(row.get("value", ""))
    if schema == "hex":
        return _normalize_hex(value)
    if schema == "rgb":
        return _rgb_to_hex(value)
    family = str(row.get("color_family"))
    return CORE_COLOR_HEX.get(family, "#777777")


def _projection_bounds(coords: np.ndarray) -> tuple[float, float, float, float]:
    min_x = float(coords[:, 0].min())
    max_x = float(coords[:, 0].max())
    min_y = float(coords[:, 1].min())
    max_y = float(coords[:, 1].max())
    if math.isclose(min_x, max_x):
        max_x = min_x + 1.0
    if math.isclose(min_y, max_y):
        max_y = min_y + 1.0
    return min_x, max_x, min_y, max_y


def _marker_shape(*, schema: str, x: float, y: float, fill: str, stroke: str, title: str) -> str:
    if schema == "word":
        return (
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="{fill}" stroke="{stroke}" stroke-width="0.9">'
            f"<title>{html.escape(title)}</title></circle>"
        )
    if schema == "hex":
        return (
            f'<rect x="{x - 4.2:.2f}" y="{y - 4.2:.2f}" width="8.4" height="8.4" rx="1.8" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="0.9"><title>{html.escape(title)}</title></rect>'
        )
    return (
        f'<polygon points="{x:.2f},{y - 4.8:.2f} {x + 4.8:.2f},{y + 4.0:.2f} {x - 4.8:.2f},{y + 4.0:.2f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="0.9"><title>{html.escape(title)}</title></polygon>'
    )


def _select_indices(
    rows: list[dict[str, Any]],
    *,
    schemas: tuple[str, ...] | None,
    include_anchors: bool,
    include_catalog: bool,
) -> list[int]:
    selected: list[int] = []
    for index, row in enumerate(rows):
        group = str(row.get("group"))
        if group == "anchor" and not include_anchors:
            continue
        if group == "catalog" and not include_catalog:
            continue
        if schemas is not None and str(row.get("schema")) not in schemas:
            continue
        selected.append(index)
    return selected


def _center_vectors(
    rows: list[dict[str, Any]],
    values: np.ndarray,
    *,
    center_mode: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if center_mode == "global":
        mean_vector = values.mean(axis=0, keepdims=True).astype(np.float32)
        return (values - mean_vector).astype(np.float32), {"all": mean_vector.squeeze(0)}
    if center_mode != "schema":
        raise ValueError(f"Unsupported center mode: {center_mode}")
    centered = np.zeros_like(values, dtype=np.float32)
    mean_vectors: dict[str, np.ndarray] = {}
    schemas = sorted({str(row.get("schema")) for row in rows})
    for schema in schemas:
        indices = [index for index, row in enumerate(rows) if str(row.get("schema")) == schema]
        schema_values = values[np.array(indices, dtype=np.int64)]
        mean_vector = schema_values.mean(axis=0, keepdims=True).astype(np.float32)
        centered[np.array(indices, dtype=np.int64)] = (schema_values - mean_vector).astype(np.float32)
        mean_vectors[schema] = mean_vector.squeeze(0)
    return centered, mean_vectors


def _torch_project(values: np.ndarray, *, device: str, projection_dim: int) -> tuple[np.ndarray, list[float]]:
    row_count = int(values.shape[0])
    feature_count = int(values.shape[1])
    target_dim = max(2, min(int(projection_dim), row_count, feature_count))

    def _project(runtime_device: str) -> tuple[np.ndarray, list[float]]:
        tensor = torch.from_numpy(values).to(runtime_device).float()
        total_variance = float((tensor.pow(2).sum() / max(row_count - 1, 1)).item()) if row_count > 1 else 0.0
        if row_count <= 1 or feature_count == 0:
            return np.zeros((row_count, 2), dtype=np.float32), [0.0, 0.0]
        u, singular_values, _ = torch.pca_lowrank(tensor, q=target_dim, center=False)
        coords = (u[:, :target_dim] * singular_values[:target_dim]).detach().cpu().numpy().astype(np.float32)
        explained = (singular_values[:target_dim] ** 2) / max(row_count - 1, 1)
        if total_variance > 1e-8:
            ratios = [float(value.item()) for value in (explained / total_variance)]
        else:
            ratios = [0.0 for _ in range(target_dim)]
        if coords.shape[1] < 2:
            padded = np.zeros((coords.shape[0], 2), dtype=np.float32)
            padded[:, : coords.shape[1]] = coords
            coords = padded
        return coords[:, :2].astype(np.float32), ratios[:2] + [0.0] * max(0, 2 - len(ratios))

    try:
        return _project(device)
    except RuntimeError:
        if device == "cpu":
            raise
        return _project("cpu")


def _write_svg(
    path: Path,
    *,
    coords: np.ndarray,
    rows: list[dict[str, Any]],
    title: str,
    subtitle: str,
) -> None:
    width = 960.0
    height = 720.0
    margin = 72.0
    inner_width = width - 2.0 * margin
    inner_height = height - 2.0 * margin
    min_x, max_x, min_y, max_y = _projection_bounds(coords)

    def project_x(value: float) -> float:
        return margin + ((value - min_x) / (max_x - min_x)) * inner_width

    def project_y(value: float) -> float:
        return height - margin - ((value - min_y) / (max_y - min_y)) * inner_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" viewBox="0 0 {int(width)} {int(height)}">',
        '<rect width="100%" height="100%" fill="#fbfaf7" />',
        f'<text x="{margin}" y="34" font-family="Helvetica, Arial, sans-serif" font-size="24" fill="#111111">{html.escape(title)}</text>',
        f'<text x="{margin}" y="58" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#555555">{html.escape(subtitle)}</text>',
        f'<rect x="{margin}" y="{margin}" width="{inner_width}" height="{inner_height}" fill="none" stroke="#bbb5ab" stroke-width="1" />',
        f'<text x="{width / 2:.0f}" y="{height - 18:.0f}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#555555">PC1</text>',
        f'<text x="22" y="{height / 2:.0f}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#555555" transform="rotate(-90 22 {height / 2:.0f})">PC2</text>',
    ]
    for row, coord in zip(rows, coords.tolist(), strict=True):
        title_text = (
            f'{row.get("record_id")} | schema={row.get("schema")} | family={row.get("color_family")} '
            f'| value={row.get("value")}'
        )
        lines.append(
            _marker_shape(
                schema=str(row.get("schema")),
                x=project_x(float(coord[0])),
                y=project_y(float(coord[1])),
                fill=_display_color(row),
                stroke=FORMAT_STROKES.get(str(row.get("schema")), "#222222"),
                title=title_text,
            )
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _analysis_specs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    schemas = tuple(sorted({str(row.get("schema")) for row in rows}))
    specs: list[dict[str, Any]] = [
        {
            "key": "all_global",
            "center_mode": "global",
            "schemas": None,
            "title": "All inputs, global mean centered",
        }
    ]
    if len(schemas) > 1:
        specs.append(
            {
                "key": "all_schema_centered",
                "center_mode": "schema",
                "schemas": None,
                "title": "All inputs, centered within input method",
            }
        )
    for schema in schemas:
        specs.append(
            {
                "key": f"schema_{schema}",
                "center_mode": "global",
                "schemas": (schema,),
                "title": f"{schema.upper()} inputs only, mean centered",
            }
        )
    return specs


def _write_index_html(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    sections: list[str] = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Geometry PCA Report</title>",
        "<style>",
        "body { font-family: Helvetica, Arial, sans-serif; background: #f5f2ea; color: #181818; margin: 24px; }",
        "h1, h2 { margin: 0 0 12px 0; }",
        "h2 { margin-top: 28px; }",
        ".card { background: #fffdf8; border: 1px solid #ddd6c9; padding: 16px; margin: 18px 0; border-radius: 12px; }",
        ".meta { color: #5b5449; font-size: 14px; margin-bottom: 10px; }",
        "img { width: 100%; max-width: 960px; border: 1px solid #e2dbcf; background: white; }",
        "code { background: #efe9dc; padding: 2px 4px; border-radius: 4px; }",
        "</style></head><body>",
        "<h1>Geometry PCA Report</h1>",
    ]
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["layer"]), []).append(row)
    for layer in sorted(grouped):
        sections.append(f"<h2>Layer {layer}</h2>")
        for row in sorted(grouped[layer], key=lambda item: str(item["analysis_key"])):
            image_path = Path(f"layer_{layer:02d}") / str(row["analysis_key"]) / "pca.svg"
            sections.append("<div class='card'>")
            sections.append(f"<div class='meta'><strong>{html.escape(str(row['analysis_title']))}</strong></div>")
            sections.append(
                f"<div class='meta'>points={int(row['point_count'])} | "
                f"PC1={float(row['pc1_variance']):.1%} | PC2={float(row['pc2_variance']):.1%}</div>"
            )
            sections.append(f"<img src='{image_path.as_posix()}' alt='{html.escape(str(row['analysis_title']))}'>")
            sections.append("</div>")
    sections.append("</body></html>")
    (output_dir / "index.html").write_text("\n".join(sections), encoding="utf-8")


def run_report(
    *,
    geometry_dir: Path,
    output_dir: Path,
    layers: tuple[int, ...] | None,
    projection_dim: int,
    include_anchors: bool,
    include_catalog: bool,
    device: str,
) -> dict[str, Any]:
    geometry_dir = geometry_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(geometry_dir / "panel.jsonl")
    layer_values = _available_layers(geometry_dir) if layers is None else layers
    analysis_specs = _analysis_specs(rows)
    layer_summary_rows: list[dict[str, Any]] = []

    if device == "auto":
        runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        runtime_device = device

    for layer in layer_values:
        encoded = np.load(geometry_dir / f"layer_{layer:02d}" / "encoded_features.npy", mmap_mode="r")
        for spec in analysis_specs:
            selected_indices = _select_indices(
                rows,
                schemas=spec["schemas"],
                include_anchors=include_anchors,
                include_catalog=include_catalog,
            )
            if len(selected_indices) < 2:
                continue
            selected_rows = [rows[index] for index in selected_indices]
            selected_values = np.asarray(encoded[np.array(selected_indices, dtype=np.int64)], dtype=np.float32)
            centered, mean_vectors = _center_vectors(
                selected_rows,
                selected_values,
                center_mode=str(spec["center_mode"]),
            )
            coords, variance = _torch_project(centered, device=runtime_device, projection_dim=projection_dim)
            analysis_dir = output_dir / f"layer_{layer:02d}" / str(spec["key"])
            analysis_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                analysis_dir / "mean_feature_vectors.npz",
                **{key: value.astype(np.float32) for key, value in mean_vectors.items()},
            )
            point_rows = [
                {
                    "analysis_key": str(spec["key"]),
                    "color_family": row.get("color_family"),
                    "display_color": _display_color(row),
                    "group": row.get("group"),
                    "pc1": float(coord[0]),
                    "pc2": float(coord[1]),
                    "record_id": row.get("record_id"),
                    "schema": row.get("schema"),
                    "value": row.get("value"),
                }
                for row, coord in zip(selected_rows, coords.tolist(), strict=True)
            ]
            _write_jsonl(analysis_dir / "pca_points.jsonl", point_rows)
            subtitle = (
                f"PC1 {variance[0]:.1%} | PC2 {variance[1]:.1%} | "
                f"points={len(selected_rows)} | centered={spec['center_mode']}"
            )
            _write_svg(
                analysis_dir / "pca.svg",
                coords=coords,
                rows=selected_rows,
                title=f"Layer {layer} | {spec['title']}",
                subtitle=subtitle,
            )
            summary_row = {
                "analysis_key": str(spec["key"]),
                "analysis_title": str(spec["title"]),
                "center_mode": str(spec["center_mode"]),
                "layer": int(layer),
                "mean_vector_keys": sorted(mean_vectors.keys()),
                "pc1_variance": float(variance[0]),
                "pc2_variance": float(variance[1]),
                "point_count": len(selected_rows),
                "schema_filter": None if spec["schemas"] is None else list(spec["schemas"]),
            }
            _write_json(analysis_dir / "summary.json", summary_row)
            layer_summary_rows.append(summary_row)

    _write_jsonl(output_dir / "layer_summary.jsonl", layer_summary_rows)
    summary = {
        "analysis_keys": [str(spec["key"]) for spec in analysis_specs],
        "geometry_dir": str(geometry_dir),
        "layers": list(layer_values),
        "output_dir": str(output_dir),
        "projection_dim": int(projection_dim),
    }
    _write_json(output_dir / "summary.json", summary)
    _write_index_html(output_dir, layer_summary_rows)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a PCA report from an existing SAE geometry run.")
    parser.add_argument("--geometry-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--layers")
    parser.add_argument("--projection-dim", type=int, default=8)
    parser.add_argument("--no-anchors", action="store_true")
    parser.add_argument("--no-catalog", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_report(
        geometry_dir=args.geometry_dir,
        output_dir=args.output_dir,
        layers=_parse_layers(args.layers),
        projection_dim=args.projection_dim,
        include_anchors=not args.no_anchors,
        include_catalog=not args.no_catalog,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
