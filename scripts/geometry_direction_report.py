#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold

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
WORD_HEX_OVERRIDES: dict[str, str] = {
    "alabaster": "#f2f0e6",
    "amaranth": "#e52b50",
    "amber": "#ffbf00",
    "amethyst": "#9966cc",
    "apricot": "#fbceb1",
    "aqua": "#00ffff",
    "aquamarine": "#7fffd4",
    "ash": "#b2beb5",
    "auburn": "#6d351a",
    "azure": "#007fff",
    "beige": "#f5f5dc",
    "bisque": "#ffe4c4",
    "bistre": "#3d2b1f",
    "blond": "#faf0be",
    "blush": "#f4c2c2",
    "bone": "#e3dac9",
    "brass": "#b5a642",
    "brick": "#a64b2a",
    "bronze": "#cd7f32",
    "buff": "#f0dc82",
    "burgundy": "#800020",
    "butter": "#fff1a8",
    "camel": "#c19a6b",
    "canary": "#ffff99",
    "caramel": "#c68e17",
    "carmine": "#960018",
    "carnation": "#ffa6c9",
    "celadon": "#ace1af",
    "celeste": "#b2ffff",
    "cerise": "#de3163",
    "cerulean": "#007ba7",
    "champagne": "#f7e7ce",
    "charcoal": "#36454f",
    "chartreuse": "#7fff00",
    "cherry": "#d2042d",
    "chestnut": "#954535",
    "chocolate": "#7b3f00",
    "cinnabar": "#e34234",
    "cinnamon": "#d2691e",
    "citrine": "#e4d00a",
    "claret": "#7f1734",
    "clay": "#b66a50",
    "cobalt": "#0047ab",
    "copper": "#b87333",
    "coquelicot": "#ff3800",
    "coral": "#ff7f50",
    "cornflower": "#6495ed",
    "cream": "#fffdd0",
    "denim": "#1560bd",
    "ebony": "#555d50",
    "ecru": "#c2b280",
    "eggplant": "#614051",
    "fawn": "#e5aa70",
    "flax": "#eedc82",
    "flint": "#6f6a60",
    "forest": "#228b22",
    "fuchsia": "#ff00ff",
    "fulvous": "#e48400",
    "garnet": "#733635",
    "ginger": "#b06500",
    "glaucous": "#6082b6",
    "golden": "#daa520",
    "graphite": "#383838",
    "gunmetal": "#2a3439",
    "harlequin": "#3fff00",
    "hazel": "#8e7618",
    "heliotrope": "#df73ff",
    "honey": "#f0b400",
    "indigo": "#4b0082",
    "ink": "#1f2a44",
    "isabelline": "#f4f0ec",
    "ivory": "#fffff0",
    "jade": "#00a86b",
    "jasper": "#d73b3e",
    "jet": "#343434",
    "khaki": "#c3b091",
    "lead": "#6c6f7f",
    "lemon": "#fff44f",
    "lilac": "#c8a2c8",
    "lime": "#bfff00",
    "loden": "#4c5b31",
    "mahogany": "#c04000",
    "maize": "#fbec5d",
    "malachite": "#0bda51",
    "maroon": "#800000",
    "mauve": "#e0b0ff",
    "melon": "#febaad",
    "midnight": "#191970",
    "mint": "#98ff98",
    "moss": "#8a9a5b",
    "mulberry": "#c54b8c",
    "mustard": "#e1ad01",
    "navy": "#000080",
    "nickel": "#727472",
    "ocher": "#cc7722",
    "ochre": "#cc7722",
    "olive": "#808000",
    "onyx": "#353839",
    "opal": "#a8c3bc",
    "orchid": "#da70d6",
    "peach": "#ffe5b4",
    "pearl": "#eae0c8",
    "peridot": "#e6e200",
    "periwinkle": "#ccccff",
    "pewter": "#96a8a1",
    "pine": "#234f1e",
    "pistachio": "#93c572",
    "platinum": "#e5e4e2",
    "plum": "#8e4585",
    "puce": "#cc8899",
    "raisin": "#6f2d5c",
    "raspberry": "#e30b5d",
    "raven": "#2b2b2b",
    "rose": "#ff007f",
    "rosewood": "#65000b",
    "rosy": "#f4c2c2",
    "royal": "#4169e1",
    "ruby": "#e0115f",
    "rufous": "#a81c07",
    "russet": "#80461b",
    "sable": "#6e403c",
    "saffron": "#f4c430",
    "salmon": "#fa8072",
    "sand": "#c2b280",
    "sangria": "#92000a",
    "sapphire": "#0f52ba",
    "sepia": "#704214",
    "shamrock": "#009e60",
    "sienna": "#882d17",
    "silver": "#c0c0c0",
    "sky": "#87ceeb",
    "slate": "#708090",
    "smoke": "#738276",
    "snow": "#fffafa",
    "sorrel": "#9d7f61",
    "steel": "#4682b4",
    "stone": "#928e85",
    "straw": "#e4d96f",
    "strawberry": "#fc5a8d",
    "sulfur": "#e9ff6a",
    "tan": "#d2b48c",
    "tangerine": "#f28500",
    "tawny": "#cd5700",
    "terracotta": "#e2725b",
    "thistle": "#d8bfd8",
    "tomato": "#ff6347",
    "topaz": "#ffc87c",
    "turquoise": "#40e0d0",
    "ultramarine": "#120a8f",
    "umber": "#635147",
    "vanilla": "#f3e5ab",
    "verdigris": "#43b3ae",
    "vermilion": "#e34234",
    "viridian": "#40826d",
    "walnut": "#773f1a",
    "wheat": "#f5deb3",
    "wine": "#722f37",
    "wisteria": "#c9a0dc",
    "xanthic": "#eeed09",
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
        layers.append(int(path.parent.name.split("_", 1)[1]))
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


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    normalized = _normalize_hex(value)
    return (int(normalized[1:3], 16), int(normalized[3:5], 16), int(normalized[5:7], 16))


def _nearest_core_family(hex_value: str) -> str:
    red, green, blue = _hex_to_rgb(hex_value)
    best_family = "gray"
    best_distance = float("inf")
    for family, family_hex in CORE_COLOR_HEX.items():
        family_red, family_green, family_blue = _hex_to_rgb(family_hex)
        distance = (
            (red - family_red) ** 2 + (green - family_green) ** 2 + (blue - family_blue) ** 2
        )
        if distance < best_distance:
            best_distance = float(distance)
            best_family = family
    return best_family


def _resolved_hex(row: dict[str, Any]) -> str:
    schema = str(row.get("schema"))
    value = str(row.get("value", "")).strip().lower()
    if schema == "hex":
        return _normalize_hex(value)
    if schema == "rgb":
        return _rgb_to_hex(value)
    if value in WORD_HEX_OVERRIDES:
        return WORD_HEX_OVERRIDES[value]
    family = str(row.get("color_family"))
    if family in CORE_COLOR_HEX:
        return CORE_COLOR_HEX[family]
    return "#777777"


def _resolved_family(row: dict[str, Any]) -> str:
    family = str(row.get("color_family"))
    if family in CORE_COLOR_HEX:
        return family
    return _nearest_core_family(_resolved_hex(row))


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


def _write_scatter_svg(
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
    ]
    for row, coord in zip(rows, coords.tolist(), strict=True):
        title_text = (
            f'{row.get("record_id")} | schema={row.get("schema")} | family={row.get("resolved_family")} '
            f'| value={row.get("value")} | hex={row.get("resolved_hex")}'
        )
        lines.append(
            _marker_shape(
                schema=str(row.get("schema")),
                x=project_x(float(coord[0])),
                y=project_y(float(coord[1])),
                fill=str(row.get("resolved_hex")),
                stroke=FORMAT_STROKES.get(str(row.get("schema")), "#222222"),
                title=title_text,
            )
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


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
    centered = np.zeros_like(values, dtype=np.float32)
    mean_vectors: dict[str, np.ndarray] = {}
    for schema in sorted({str(row.get("schema")) for row in rows}):
        indices = [index for index, row in enumerate(rows) if str(row.get("schema")) == schema]
        schema_values = values[np.array(indices, dtype=np.int64)]
        mean_vector = schema_values.mean(axis=0, keepdims=True).astype(np.float32)
        centered[np.array(indices, dtype=np.int64)] = (schema_values - mean_vector).astype(np.float32)
        mean_vectors[schema] = mean_vector.squeeze(0)
    return centered, mean_vectors


def _torch_pca(values: np.ndarray, *, device: str, projection_dim: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
    row_count = int(values.shape[0])
    feature_count = int(values.shape[1])
    target_dim = max(2, min(int(projection_dim), row_count, feature_count))

    def _project(runtime_device: str) -> tuple[np.ndarray, np.ndarray, list[float]]:
        tensor = torch.from_numpy(values).to(runtime_device).float()
        total_variance = float((tensor.pow(2).sum() / max(row_count - 1, 1)).item()) if row_count > 1 else 0.0
        if row_count <= 1 or feature_count == 0:
            zeros = np.zeros((row_count, 2), dtype=np.float32)
            return zeros, zeros, [0.0, 0.0]
        u, singular_values, v = torch.pca_lowrank(tensor, q=target_dim, center=False)
        coords = (u[:, :target_dim] * singular_values[:target_dim]).detach().cpu().numpy().astype(np.float32)
        components = v[:, :target_dim].T.detach().cpu().numpy().astype(np.float32)
        explained = (singular_values[:target_dim] ** 2) / max(row_count - 1, 1)
        ratios = [float(value.item()) for value in (explained / total_variance)] if total_variance > 1e-8 else [0.0 for _ in range(target_dim)]
        if coords.shape[1] < 2:
            padded = np.zeros((coords.shape[0], 2), dtype=np.float32)
            padded[:, : coords.shape[1]] = coords
            coords = padded
        return coords[:, :2].astype(np.float32), components, ratios[:2] + [0.0] * max(0, 2 - len(ratios))

    try:
        return _project(device)
    except RuntimeError:
        if device == "cpu":
            raise
        return _project("cpu")


def _lda_projection(values: np.ndarray, labels: list[str]) -> tuple[np.ndarray | None, dict[str, Any]]:
    if len(set(labels)) < 2 or values.shape[0] < 3:
        return None, {"class_count": len(set(labels)), "status": "skipped"}
    lda = LinearDiscriminantAnalysis(n_components=2, solver="svd")
    try:
        coords = lda.fit_transform(values, labels).astype(np.float32)
    except Exception as exc:
        return None, {"class_count": len(set(labels)), "status": f"failed: {exc}"}
    if coords.shape[1] < 2:
        padded = np.zeros((coords.shape[0], 2), dtype=np.float32)
        padded[:, : coords.shape[1]] = coords
        coords = padded
    ratios = getattr(lda, "explained_variance_ratio_", None)
    if ratios is None:
        ratio_values = [None, None]
    else:
        ratio_values = [float(ratios[index]) if index < len(ratios) else None for index in range(2)]
    return coords, {
        "class_count": len(set(labels)),
        "explained_variance_ratio": ratio_values,
        "status": "ok",
    }


def _cross_validated_probe_accuracy(values: np.ndarray, labels: list[str]) -> float | None:
    label_counts = Counter(labels)
    min_class_count = min(label_counts.values())
    if min_class_count < 2 or len(set(labels)) < 2:
        return None
    split_count = min(5, min_class_count)
    splitter = StratifiedKFold(n_splits=split_count, shuffle=True, random_state=17)
    scores: list[float] = []
    label_array = np.array(labels)
    for train_indices, test_indices in splitter.split(values, label_array):
        model = RidgeClassifier(class_weight="balanced")
        model.fit(values[train_indices], label_array[train_indices])
        predicted = model.predict(values[test_indices])
        scores.append(float((predicted == label_array[test_indices]).mean()))
    return float(sum(scores) / len(scores)) if scores else None


def _schema_transfer_accuracy(values: np.ndarray, rows: list[dict[str, Any]]) -> float | None:
    source_indices = [index for index, row in enumerate(rows) if str(row.get("schema")) == "word"]
    target_indices = [
        index for index, row in enumerate(rows) if str(row.get("schema")) in {"hex", "rgb"}
    ]
    if not source_indices or not target_indices:
        return None
    source_values = values[np.array(source_indices, dtype=np.int64)]
    source_labels = [str(rows[index]["resolved_family"]) for index in source_indices]
    target_values = values[np.array(target_indices, dtype=np.int64)]
    target_labels = np.array([str(rows[index]["resolved_family"]) for index in target_indices])
    if len(set(source_labels)) < 2:
        return None
    model = RidgeClassifier(class_weight="balanced")
    model.fit(source_values, np.array(source_labels))
    predicted = model.predict(target_values)
    return float((predicted == target_labels).mean())


def _cluster_metrics(pca_coords: np.ndarray, labels: list[str]) -> dict[str, Any]:
    family_count = len(set(labels))
    if family_count < 2 or len(labels) < family_count:
        return {"status": "skipped"}
    cluster_input = pca_coords[:, : min(2, pca_coords.shape[1])]
    model = KMeans(n_clusters=family_count, n_init=20, random_state=17)
    assignments = model.fit_predict(cluster_input)
    purity_terms: list[float] = []
    for cluster_id in sorted(set(int(value) for value in assignments)):
        cluster_labels = [labels[index] for index, value in enumerate(assignments) if int(value) == cluster_id]
        counts = Counter(cluster_labels)
        purity_terms.append(float(counts.most_common(1)[0][1] / len(cluster_labels)))
    return {
        "adjusted_rand_index": float(adjusted_rand_score(labels, assignments)),
        "cluster_count": family_count,
        "mean_cluster_purity": float(sum(purity_terms) / len(purity_terms)),
        "status": "ok",
    }


def _mean_direction_report(values: np.ndarray, labels: list[str]) -> dict[str, Any]:
    total_mean = values.mean(axis=0)
    rows: list[dict[str, Any]] = []
    label_array = np.array(labels)
    for family in sorted(set(labels)):
        positive_values = values[label_array == family]
        negative_values = values[label_array != family]
        if positive_values.shape[0] == 0 or negative_values.shape[0] == 0:
            continue
        centroid = positive_values.mean(axis=0)
        direction = centroid - negative_values.mean(axis=0)
        positive_scores = positive_values @ direction
        negative_scores = negative_values @ direction
        pooled_std = float(np.std(np.concatenate([positive_scores, negative_scores]))) + 1e-6
        rows.append(
            {
                "centroid_norm": float(np.linalg.norm(centroid - total_mean)),
                "direction_norm": float(np.linalg.norm(direction)),
                "d_prime": float((positive_scores.mean() - negative_scores.mean()) / pooled_std),
                "family": family,
            }
        )
    rows.sort(key=lambda row: float(row["d_prime"]), reverse=True)
    return {
        "families": rows,
        "top_family": None if not rows else rows[0]["family"],
        "top_family_d_prime": None if not rows else rows[0]["d_prime"],
    }


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


def _decorate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decorated: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        updated["resolved_hex"] = _resolved_hex(row)
        updated["resolved_family"] = _resolved_family(row)
        decorated.append(updated)
    return decorated


def _write_index_html(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    sections = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Geometry Direction Report</title>",
        "<style>",
        "body { font-family: Helvetica, Arial, sans-serif; background: #f5f2ea; color: #181818; margin: 24px; }",
        "h1, h2, h3 { margin: 0 0 12px 0; }",
        "h2 { margin-top: 28px; }",
        ".card { background: #fffdf8; border: 1px solid #ddd6c9; padding: 16px; margin: 18px 0; border-radius: 12px; }",
        ".meta { color: #5b5449; font-size: 14px; margin-bottom: 10px; }",
        "img { width: 100%; max-width: 960px; border: 1px solid #e2dbcf; background: white; margin-top: 10px; }",
        "code { background: #efe9dc; padding: 2px 4px; border-radius: 4px; }",
        "table { border-collapse: collapse; margin-top: 10px; }",
        "td, th { border: 1px solid #d8cfbf; padding: 6px 10px; text-align: left; }",
        "</style></head><body>",
        "<h1>Geometry Direction Report</h1>",
    ]
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["layer"]), []).append(row)
    for layer in sorted(grouped):
        sections.append(f"<h2>Layer {layer}</h2>")
        for row in sorted(grouped[layer], key=lambda item: str(item["analysis_key"])):
            base = Path(f"layer_{layer:02d}") / str(row["analysis_key"])
            sections.append("<div class='card'>")
            sections.append(f"<h3>{html.escape(str(row['analysis_title']))}</h3>")
            sections.append(
                f"<div class='meta'>points={int(row['point_count'])} | families={int(row['family_count'])} | "
                f"PC1={float(row['pc1_variance']):.1%} | PC2={float(row['pc2_variance']):.1%}</div>"
            )
            sections.append("<table>")
            sections.append("<tr><th>Technique</th><th>Result</th></tr>")
            sections.append(f"<tr><td>Family probe CV</td><td>{row.get('family_probe_accuracy')}</td></tr>")
            sections.append(f"<tr><td>Schema probe CV</td><td>{row.get('schema_probe_accuracy')}</td></tr>")
            sections.append(f"<tr><td>Word to hex/rgb transfer</td><td>{row.get('schema_transfer_accuracy')}</td></tr>")
            sections.append(f"<tr><td>Cluster purity</td><td>{row.get('cluster_purity')}</td></tr>")
            sections.append(f"<tr><td>Cluster ARI</td><td>{row.get('cluster_ari')}</td></tr>")
            sections.append(f"<tr><td>Top family direction</td><td>{row.get('top_family_direction')}</td></tr>")
            sections.append("</table>")
            sections.append(f"<img src='{(base / 'pca.svg').as_posix()}' alt='PCA plot'>")
            lda_path = base / "lda.svg"
            if lda_path.exists():
                sections.append(f"<img src='{lda_path.as_posix()}' alt='LDA plot'>")
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
    rows = _decorate_rows(_read_jsonl(geometry_dir / "panel.jsonl"))
    layer_values = _available_layers(geometry_dir) if layers is None else layers
    analysis_specs = _analysis_specs(rows)
    if device == "auto":
        runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        runtime_device = device

    summary_rows: list[dict[str, Any]] = []
    for layer in layer_values:
        encoded = np.load(geometry_dir / f"layer_{layer:02d}" / "encoded_features.npy", mmap_mode="r")
        for spec in analysis_specs:
            selected_indices = _select_indices(
                rows,
                schemas=spec["schemas"],
                include_anchors=include_anchors,
                include_catalog=include_catalog,
            )
            if len(selected_indices) < 3:
                continue
            selected_rows = [rows[index] for index in selected_indices]
            selected_values = np.asarray(encoded[np.array(selected_indices, dtype=np.int64)], dtype=np.float32)
            centered, mean_vectors = _center_vectors(
                selected_rows,
                selected_values,
                center_mode=str(spec["center_mode"]),
            )
            family_labels = [str(row["resolved_family"]) for row in selected_rows]
            schema_labels = [str(row["schema"]) for row in selected_rows]
            pca_coords, _, pca_variance = _torch_pca(centered, device=runtime_device, projection_dim=projection_dim)

            analysis_dir = output_dir / f"layer_{layer:02d}" / str(spec["key"])
            analysis_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                analysis_dir / "mean_feature_vectors.npz",
                **{key: value.astype(np.float32) for key, value in mean_vectors.items()},
            )
            point_rows = [
                {
                    "analysis_key": str(spec["key"]),
                    "pc1": float(coord[0]),
                    "pc2": float(coord[1]),
                    "record_id": row.get("record_id"),
                    "resolved_family": row.get("resolved_family"),
                    "resolved_hex": row.get("resolved_hex"),
                    "schema": row.get("schema"),
                    "value": row.get("value"),
                }
                for row, coord in zip(selected_rows, pca_coords.tolist(), strict=True)
            ]
            _write_jsonl(analysis_dir / "pca_points.jsonl", point_rows)
            _write_scatter_svg(
                analysis_dir / "pca.svg",
                coords=pca_coords,
                rows=selected_rows,
                title=f"Layer {layer} | {spec['title']} | PCA",
                subtitle=f"PC1 {pca_variance[0]:.1%} | PC2 {pca_variance[1]:.1%} | points={len(selected_rows)}",
            )

            lda_coords, lda_summary = _lda_projection(centered, family_labels)
            if lda_coords is not None:
                _write_scatter_svg(
                    analysis_dir / "lda.svg",
                    coords=lda_coords,
                    rows=selected_rows,
                    title=f"Layer {layer} | {spec['title']} | LDA",
                    subtitle=f"families={len(set(family_labels))} | status={lda_summary['status']}",
                )

            family_probe_accuracy = _cross_validated_probe_accuracy(centered, family_labels)
            schema_probe_accuracy = _cross_validated_probe_accuracy(centered, schema_labels)
            transfer_accuracy = _schema_transfer_accuracy(centered, selected_rows)
            cluster_summary = _cluster_metrics(pca_coords, family_labels)
            direction_summary = _mean_direction_report(centered, family_labels)

            family_counts = Counter(family_labels)
            summary_row = {
                "analysis_key": str(spec["key"]),
                "analysis_title": str(spec["title"]),
                "center_mode": str(spec["center_mode"]),
                "cluster_ari": cluster_summary.get("adjusted_rand_index"),
                "cluster_purity": cluster_summary.get("mean_cluster_purity"),
                "family_count": len(family_counts),
                "family_counts": dict(sorted(family_counts.items())),
                "family_probe_accuracy": family_probe_accuracy,
                "layer": int(layer),
                "lda_status": lda_summary.get("status"),
                "mean_vector_keys": sorted(mean_vectors.keys()),
                "pc1_variance": float(pca_variance[0]),
                "pc2_variance": float(pca_variance[1]),
                "point_count": len(selected_rows),
                "schema_filter": None if spec["schemas"] is None else list(spec["schemas"]),
                "schema_probe_accuracy": schema_probe_accuracy,
                "schema_transfer_accuracy": transfer_accuracy,
                "top_family_direction": direction_summary.get("top_family"),
                "top_family_direction_d_prime": direction_summary.get("top_family_d_prime"),
            }
            _write_json(analysis_dir / "summary.json", summary_row)
            _write_json(analysis_dir / "lda_summary.json", lda_summary)
            _write_json(analysis_dir / "cluster_summary.json", cluster_summary)
            _write_json(analysis_dir / "mean_direction_summary.json", direction_summary)
            summary_rows.append(summary_row)

    _write_jsonl(output_dir / "layer_summary.jsonl", summary_rows)
    summary = {
        "analysis_keys": [str(spec["key"]) for spec in analysis_specs],
        "geometry_dir": str(geometry_dir),
        "layers": list(layer_values),
        "output_dir": str(output_dir),
        "projection_dim": int(projection_dim),
    }
    _write_json(output_dir / "summary.json", summary)
    _write_index_html(output_dir, summary_rows)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a direction-analysis report from an existing SAE geometry run.")
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
