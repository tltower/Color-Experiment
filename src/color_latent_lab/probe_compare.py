from __future__ import annotations

from pathlib import Path
from typing import Any

from .analysis_common import (
    available_activation_layers,
    read_jsonl as _read_jsonl,
    write_json as _write_json,
    write_jsonl as _write_jsonl,
)


def _require_probe_stack() -> tuple[Any, Any, Any]:
    try:
        import numpy as np  # type: ignore[import-not-found]
        from sklearn.linear_model import RidgeClassifier  # type: ignore[import-not-found]
        from sklearn.model_selection import StratifiedKFold  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "The research stack is not installed. Use `pip install -e .` in the repo environment."
        ) from exc
    return np, RidgeClassifier, StratifiedKFold
def _label_for_row(row: dict[str, Any], *, label_mode: str) -> str | None:
    if label_mode == "family":
        value = row.get("color_family")
        return None if value is None else str(value)
    if label_mode == "color_word":
        value = row.get("color_label")
        if value is not None:
            return str(value)
        if str(row.get("schema")) == "word":
            raw_value = row.get("display_label", row.get("value"))
            return None if raw_value is None else str(raw_value).strip().lower()
        return None
    raise ValueError(f"Unsupported label_mode: {label_mode}")


def _select_row_indices(
    rows: list[dict[str, Any]],
    *,
    schemas: tuple[str, ...] | None,
    include_anchors: bool,
    include_catalog: bool,
    label_mode: str,
) -> tuple[list[int], list[str]]:
    selected_indices: list[int] = []
    labels: list[str] = []
    for index, row in enumerate(rows):
        group = str(row.get("group"))
        if group == "anchor" and not include_anchors:
            continue
        if group == "catalog" and not include_catalog:
            continue
        if schemas is not None and str(row.get("schema")) not in schemas:
            continue
        label = _label_for_row(row, label_mode=label_mode)
        if label is None:
            continue
        selected_indices.append(index)
        labels.append(label)
    return selected_indices, labels


def _center_vectors(np: Any, rows: list[dict[str, Any]], values: Any, *, center_mode: str) -> Any:
    centered = values.astype(np.float32)
    if center_mode == "none":
        return centered
    if center_mode == "global":
        mean_vector = centered.mean(axis=0, keepdims=True).astype(np.float32)
        return (centered - mean_vector).astype(np.float32)
    if center_mode != "schema":
        raise ValueError(f"Unsupported center_mode: {center_mode}")
    output = np.zeros_like(centered, dtype=np.float32)
    for schema in sorted({str(row.get("schema")) for row in rows}):
        indices = [index for index, row in enumerate(rows) if str(row.get("schema")) == schema]
        schema_values = centered[np.array(indices, dtype=np.int64)]
        mean_vector = schema_values.mean(axis=0, keepdims=True).astype(np.float32)
        output[np.array(indices, dtype=np.int64)] = (schema_values - mean_vector).astype(np.float32)
    return output


def _stratified_splits(StratifiedKFold: Any, labels: list[str]) -> list[tuple[Any, Any]]:
    label_counts: dict[str, int] = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    if not label_counts:
        raise ValueError("No labeled rows selected.")
    min_class_count = min(label_counts.values())
    if min_class_count < 2:
        raise ValueError(
            "Selected labels do not repeat enough for held-out probe evaluation. "
            "For exact color-word probes, mirror the catalog across input formats so each color label appears multiple times."
        )
    split_count = min(5, min_class_count)
    splitter = StratifiedKFold(n_splits=split_count, shuffle=True, random_state=17)
    indices = list(range(len(labels)))
    import numpy as np  # local import to keep helper lightweight

    label_array = np.array(labels)
    return list(splitter.split(np.array(indices).reshape(-1, 1), label_array))


def _cv_accuracy(
    *,
    np: Any,
    RidgeClassifier: Any,
    values: Any,
    labels: list[str],
    splits: list[tuple[Any, Any]],
) -> tuple[float, list[float]]:
    label_array = np.array(labels)
    scores: list[float] = []
    for train_indices, test_indices in splits:
        model = RidgeClassifier(class_weight="balanced")
        model.fit(values[train_indices], label_array[train_indices])
        predicted = model.predict(values[test_indices])
        scores.append(float((predicted == label_array[test_indices]).mean()))
    return float(sum(scores) / len(scores)), scores


def _write_accuracy_curve_svg(path: Path, *, layer_rows: list[dict[str, Any]]) -> None:
    if not layer_rows:
        return
    width = 860.0
    height = 420.0
    margin = 64.0
    inner_width = width - 2 * margin
    inner_height = height - 2 * margin
    layers = [int(row["layer"]) for row in layer_rows]
    min_layer = min(layers)
    max_layer = max(layers)
    if min_layer == max_layer:
        max_layer = min_layer + 1

    def project_x(layer: float) -> float:
        return margin + ((layer - min_layer) / (max_layer - min_layer)) * inner_width

    def project_y(score: float) -> float:
        bounded = max(0.0, min(1.0, score))
        return height - margin - bounded * inner_height

    def polyline(values: list[tuple[float, float]], color: str) -> str:
        points = " ".join(f"{project_x(layer):.2f},{project_y(score):.2f}" for layer, score in values)
        return f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{points}" />'

    residual_points = [(float(row["layer"]), float(row["residual_accuracy"])) for row in layer_rows]
    sae_points = [(float(row["layer"]), float(row["sae_accuracy"])) for row in layer_rows]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" viewBox="0 0 {int(width)} {int(height)}">',
        '<rect width="100%" height="100%" fill="#fbfaf7" />',
        '<text x="64" y="34" font-family="Helvetica, Arial, sans-serif" font-size="24" fill="#111111">Residual vs SAE probe accuracy</text>',
        '<text x="64" y="58" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#555555">Held-out multiclass accuracy by layer</text>',
        f'<rect x="{margin}" y="{margin}" width="{inner_width}" height="{inner_height}" fill="none" stroke="#bbb5ab" stroke-width="1" />',
        polyline(residual_points, "#2b63c9"),
        polyline(sae_points, "#c93a3a"),
    ]
    for row in layer_rows:
        for key, color in (("residual_accuracy", "#2b63c9"), ("sae_accuracy", "#c93a3a")):
            x = project_x(float(row["layer"]))
            y = project_y(float(row[key]))
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="{color}" />')
            lines.append(
                f'<text x="{x:.2f}" y="{y - 10:.2f}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="{color}">{float(row[key]):.3f}</text>'
            )
    lines.extend(
        [
            '<text x="720" y="80" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#2b63c9">Residual stream</text>',
            '<text x="720" y="98" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#c93a3a">SAE code</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_probe_comparison(
    *,
    geometry_dir: Path,
    output_dir: Path,
    layers: tuple[int, ...] | None = None,
    label_mode: str = "family",
    schema_filter: tuple[str, ...] | None = None,
    include_anchors: bool = True,
    include_catalog: bool = True,
    center_mode: str = "schema",
) -> dict[str, Any]:
    np, RidgeClassifier, StratifiedKFold = _require_probe_stack()
    geometry_dir = geometry_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_rows = _read_jsonl(geometry_dir / "panel.jsonl")
    selected_indices, labels = _select_row_indices(
        panel_rows,
        schemas=schema_filter,
        include_anchors=include_anchors,
        include_catalog=include_catalog,
        label_mode=label_mode,
    )
    selected_rows = [panel_rows[index] for index in selected_indices]
    splits = _stratified_splits(StratifiedKFold, labels)
    resolved_layers = available_activation_layers(geometry_dir) if layers is None else layers

    layer_summary_rows: list[dict[str, Any]] = []
    for layer in resolved_layers:
        residual = np.load(geometry_dir / "activations" / f"layer_{layer:02d}.npy").astype(np.float32)
        sae = np.load(geometry_dir / f"layer_{layer:02d}" / "encoded_features.npy").astype(np.float32)
        selected_residual = residual[np.array(selected_indices, dtype=np.int64)]
        selected_sae = sae[np.array(selected_indices, dtype=np.int64)]
        centered_residual = _center_vectors(np, selected_rows, selected_residual, center_mode=center_mode)
        centered_sae = _center_vectors(np, selected_rows, selected_sae, center_mode=center_mode)
        residual_accuracy, residual_fold_scores = _cv_accuracy(
            np=np,
            RidgeClassifier=RidgeClassifier,
            values=centered_residual,
            labels=labels,
            splits=splits,
        )
        sae_accuracy, sae_fold_scores = _cv_accuracy(
            np=np,
            RidgeClassifier=RidgeClassifier,
            values=centered_sae,
            labels=labels,
            splits=splits,
        )
        row = {
            "center_mode": center_mode,
            "label_mode": label_mode,
            "layer": int(layer),
            "point_count": len(selected_rows),
            "residual_accuracy": residual_accuracy,
            "residual_fold_scores": residual_fold_scores,
            "sae_accuracy": sae_accuracy,
            "sae_fold_scores": sae_fold_scores,
            "sae_minus_residual": float(sae_accuracy - residual_accuracy),
            "schema_filter": None if schema_filter is None else list(schema_filter),
        }
        _write_json(output_dir / f"layer_{layer:02d}.summary.json", row)
        layer_summary_rows.append(row)

    layer_summary_rows.sort(key=lambda row: int(row["layer"]))
    _write_jsonl(output_dir / "layer_summary.jsonl", layer_summary_rows)
    _write_accuracy_curve_svg(output_dir / "accuracy_curve.svg", layer_rows=layer_summary_rows)
    best_residual = max(layer_summary_rows, key=lambda row: float(row["residual_accuracy"]))
    best_sae = max(layer_summary_rows, key=lambda row: float(row["sae_accuracy"]))
    summary = {
        "best_residual_accuracy": float(best_residual["residual_accuracy"]),
        "best_residual_layer": int(best_residual["layer"]),
        "best_sae_accuracy": float(best_sae["sae_accuracy"]),
        "best_sae_layer": int(best_sae["layer"]),
        "center_mode": center_mode,
        "geometry_dir": str(geometry_dir),
        "label_mode": label_mode,
        "layers": [int(row["layer"]) for row in layer_summary_rows],
        "point_count": len(selected_rows),
        "schema_filter": None if schema_filter is None else list(schema_filter),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


__all__ = ["run_probe_comparison"]
