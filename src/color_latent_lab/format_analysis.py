from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

from .color_formats import DEFAULT_FORMATS, FAMILY_PALETTE, FORMAT_STROKES
from .run_support import HeartbeatRecorder, _write_jsonl
from .workflow_common import _mean_or_none


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
    shared_words = sorted(set(source_index) & set(target_index) & set(consensus_by_word))
    if len(shared_words) < 4:
        return None
    source_features = np.stack([source_matrix[source_index[word]] for word in shared_words]).astype(np.float32)
    target_features = np.stack([target_matrix[target_index[word]] for word in shared_words]).astype(np.float32)
    labels = [consensus_by_word[word] for word in shared_words]
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


__all__ = [
    "_build_layer_analysis",
    "_fit_probe_accuracy",
    "_fit_transfer_accuracy",
    "_fit_within_schema_accuracy",
    "_mean_or_none",
]
