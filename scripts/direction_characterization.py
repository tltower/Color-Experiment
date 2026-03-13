#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from color_latent_lab.analysis_common import (
    available_direction_layers as _available_layers,
    cosine_similarity_matrix as _cosine_similarity_matrix,
    parse_layers as _parse_layers,
    read_json as _read_json,
    read_jsonl as _read_jsonl,
    write_json as _write_json,
)
from color_latent_lab.color_palette import CORE_COLOR_HEX
from color_latent_lab.sae_geometry import CORE_COLOR_FAMILIES, _load_direction


def _effective_dimensionality(direction_matrix: np.ndarray) -> dict[str, Any]:
    centered = direction_matrix - direction_matrix.mean(axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values**2
    total = float(variance.sum())
    ratios = [float(value / total) for value in variance] if total > 0 else [0.0 for value in variance]
    cumulative = 0.0
    components_90 = 0
    for index, ratio in enumerate(ratios, start=1):
        cumulative += ratio
        if cumulative >= 0.9:
            components_90 = index
            break
    if components_90 == 0:
        components_90 = len(ratios)
    return {
        "components_for_90_percent": components_90,
        "explained_variance_ratio": ratios,
        "singular_values": [float(value) for value in singular_values],
    }


def _pca_2d(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    centered = values - values.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    variance = singular_values**2
    total = float(variance.sum())
    ratios = [float(value / total) for value in variance[:2]] if total > 0 else [0.0, 0.0]
    if coords.shape[1] < 2:
        padded = np.zeros((coords.shape[0], 2), dtype=np.float32)
        padded[:, : coords.shape[1]] = coords
        coords = padded
    return coords.astype(np.float32), ratios


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


def _write_cosine_heatmap(path: Path, *, families: list[str], matrix: np.ndarray, title: str) -> None:
    cell = 42
    margin = 120
    width = margin + cell * len(families) + 24
    height = margin + cell * len(families) + 24

    def color_for(value: float) -> str:
        normalized = max(-1.0, min(1.0, value))
        if normalized >= 0:
            red = int(255 - normalized * 55)
            green = int(245 - normalized * 155)
            blue = int(245 - normalized * 155)
        else:
            red = int(245 + normalized * 155)
            green = int(245 + normalized * 155)
            blue = int(255 - normalized * 55)
        return f"#{red:02x}{green:02x}{blue:02x}"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7" />',
        f'<text x="24" y="34" font-family="Helvetica, Arial, sans-serif" font-size="24" fill="#111111">{html.escape(title)}</text>',
    ]
    for index, family in enumerate(families):
        x = margin + index * cell
        y = margin - 12
        lines.append(
            f'<text x="{x + cell / 2:.1f}" y="{y:.1f}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#222222">{family}</text>'
        )
        lines.append(
            f'<text x="{margin - 12:.1f}" y="{margin + index * cell + cell / 2 + 4:.1f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#222222">{family}</text>'
        )
    for row_index, family in enumerate(families):
        for col_index, _other in enumerate(families):
            value = float(matrix[row_index, col_index])
            x = margin + col_index * cell
            y = margin + row_index * cell
            lines.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color_for(value)}" stroke="#ddd6c9" stroke-width="1" />'
            )
            lines.append(
                f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2 + 4:.1f}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="10" fill="#111111">{value:.2f}</text>'
            )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_direction_pca_svg(
    path: Path,
    *,
    families: list[str],
    coords: np.ndarray,
    pc_variance: list[float],
    title: str,
) -> None:
    width = 960.0
    height = 720.0
    margin = 80.0
    inner_width = width - 2 * margin
    inner_height = height - 2 * margin
    min_x, max_x, min_y, max_y = _projection_bounds(coords)

    def project_x(value: float) -> float:
        return margin + ((value - min_x) / (max_x - min_x)) * inner_width

    def project_y(value: float) -> float:
        return height - margin - ((value - min_y) / (max_y - min_y)) * inner_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" viewBox="0 0 {int(width)} {int(height)}">',
        '<rect width="100%" height="100%" fill="#fbfaf7" />',
        f'<text x="{margin}" y="34" font-family="Helvetica, Arial, sans-serif" font-size="24" fill="#111111">{html.escape(title)}</text>',
        f'<text x="{margin}" y="58" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#555555">PC1 {pc_variance[0]:.1%} | PC2 {pc_variance[1]:.1%}</text>',
        f'<rect x="{margin}" y="{margin}" width="{inner_width}" height="{inner_height}" fill="none" stroke="#bbb5ab" stroke-width="1" />',
    ]
    for family, coord in zip(families, coords.tolist(), strict=True):
        x = project_x(float(coord[0]))
        y = project_y(float(coord[1]))
        fill = CORE_COLOR_HEX.get(family, "#777777")
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="8" fill="{fill}" stroke="#111111" stroke-width="1.2" />')
        lines.append(
            f'<text x="{x + 10:.2f}" y="{y - 10:.2f}" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#111111">{family}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_feature_scores(layer_dir: Path) -> dict[int, dict[str, Any]]:
    scores_path = layer_dir / "feature_scores.jsonl"
    if not scores_path.exists():
        return {}
    rows = _read_jsonl(scores_path)
    return {int(row["feature"]): row for row in rows}


def _feature_role(*, color_eta_squared: Any, format_eta_squared: Any) -> str:
    if color_eta_squared is None or format_eta_squared is None:
        return "unscored"
    color_value = float(color_eta_squared)
    format_value = float(format_eta_squared)
    if color_value >= 0.2 and format_value <= 0.1:
        return "color_selective"
    if format_value > color_value:
        return "format_contaminated"
    return "mixed"


def _top_feature_prompts(
    *,
    encoded_features: np.ndarray | None,
    panel_rows: list[dict[str, Any]] | None,
    feature: int,
    top_n: int,
) -> list[dict[str, Any]]:
    if encoded_features is None or panel_rows is None or feature >= int(encoded_features.shape[1]):
        return []
    column = encoded_features[:, feature]
    ranked = np.argsort(-column)[:top_n]
    exemplars: list[dict[str, Any]] = []
    for index in ranked.tolist():
        row = panel_rows[int(index)]
        exemplars.append(
            {
                "activation": float(column[int(index)]),
                "color_family": row.get("color_family"),
                "color_label": row.get("color_label"),
                "prompt": row.get("prompt"),
                "schema": row.get("schema"),
                "value": row.get("value"),
            }
        )
    return exemplars


def _attribute_directions(
    layer_dir: Path,
    *,
    top_k: int,
    encoded_features: np.ndarray | None = None,
    panel_rows: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    rankings_path = layer_dir / "family_feature_rankings.json"
    if not rankings_path.exists():
        return {}
    rankings = _read_json(rankings_path)
    feature_scores = _load_feature_scores(layer_dir)
    attribution: dict[str, list[dict[str, Any]]] = {}
    for family, rows in rankings.items():
        enriched_rows: list[dict[str, Any]] = []
        for row in rows[:top_k]:
            feature = int(row["feature"])
            scores = feature_scores.get(feature, {})
            enriched_rows.append(
                {
                    "feature": feature,
                    "delta": float(row["delta"]),
                    "positive_mean": float(row["positive_mean"]),
                    "negative_mean": float(row["negative_mean"]),
                    "color_eta_squared": scores.get("color_eta_squared"),
                    "format_eta_squared": scores.get("format_eta_squared"),
                    "invariant_score": scores.get("invariant_score"),
                    "feature_role": _feature_role(
                        color_eta_squared=scores.get("color_eta_squared"),
                        format_eta_squared=scores.get("format_eta_squared"),
                    ),
                    "top_prompts": _top_feature_prompts(
                        encoded_features=encoded_features,
                        panel_rows=panel_rows,
                        feature=feature,
                        top_n=4,
                    ),
                }
            )
        attribution[family] = enriched_rows
    return attribution


def _warm_cool_projection(direction_rows: dict[str, np.ndarray], warm_cool_direction: np.ndarray | None) -> dict[str, float]:
    if warm_cool_direction is None:
        return {}
    norm = float(np.linalg.norm(warm_cool_direction)) + 1e-8
    unit = warm_cool_direction / norm
    return {
        family: float(np.dot(direction, unit))
        for family, direction in direction_rows.items()
    }


def _opponent_summary(families: list[str], cosine_matrix: np.ndarray) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for index, family in enumerate(families):
        scores = [
            (other_family, float(cosine_matrix[index, other_index]))
            for other_index, other_family in enumerate(families)
            if other_family != family
        ]
        most_aligned = max(scores, key=lambda item: item[1])
        most_opposed = min(scores, key=lambda item: item[1])
        rows[family] = {
            "most_aligned_family": most_aligned[0],
            "most_aligned_cosine": most_aligned[1],
            "most_opposed_family": most_opposed[0],
            "most_opposed_cosine": most_opposed[1],
        }
    return rows


def _write_direction_feature_cards(path: Path, *, feature_attribution: dict[str, list[dict[str, Any]]]) -> None:
    lines = [
        "# Direction feature cards",
        "",
        "These cards are intended as the human-inspection step before interventions.",
        "",
    ]
    for family in CORE_COLOR_FAMILIES:
        rows = feature_attribution.get(family, [])
        if not rows:
            continue
        lines.extend([f"## {family}", ""])
        for row in rows:
            lines.append(
                (
                    f"- feature `{row['feature']}` | delta `{row['delta']:.4f}` | "
                    f"role `{row['feature_role']}` | color eta `{row['color_eta_squared']}` | "
                    f"format eta `{row['format_eta_squared']}`"
                )
            )
            top_prompts = row.get("top_prompts", [])
            if top_prompts:
                prompt_bits = [
                    f"{prompt_row['schema']}:{prompt_row['value']} ({prompt_row['activation']:.3f})"
                    for prompt_row in top_prompts[:3]
                ]
                lines.append(f"  top prompts: {', '.join(prompt_bits)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_intervention_rows(intervention_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(intervention_root.rglob("summary.json")):
        try:
            payload = _read_json(summary_path)
        except json.JSONDecodeError:
            continue
        if "target_family" not in payload or "layer" not in payload:
            continue
        rows.append(
            {
                "best_alpha": payload.get("best_alpha"),
                "best_target_match_rate": payload.get("best_target_match_rate"),
                "layer": payload["layer"],
                "prompt_mode": payload.get("prompt_mode"),
                "target_family": payload["target_family"],
            }
        )
    return rows


def _intervention_correlation(
    *,
    layer: int,
    direction_summary: dict[str, Any],
    intervention_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    relevant = [row for row in intervention_rows if int(row["layer"]) == int(layer)]
    if not relevant:
        return None
    d_prime_by_family = {
        str(row["family"]): float(row["d_prime"])
        for row in direction_summary.get("families", [])
    }
    paired = [
        (d_prime_by_family[row["target_family"]], float(row["best_target_match_rate"]))
        for row in relevant
        if row["target_family"] in d_prime_by_family and row.get("best_target_match_rate") is not None
    ]
    if len(paired) < 2:
        return None
    d_primes = np.array([item[0] for item in paired], dtype=np.float32)
    steer = np.array([item[1] for item in paired], dtype=np.float32)
    corr = float(np.corrcoef(d_primes, steer)[0, 1])
    return {"correlation": corr, "pair_count": len(paired)}


def run_characterization(
    *,
    geometry_dir: Path,
    output_dir: Path,
    layers: tuple[int, ...] | None,
    top_k: int,
    intervention_root: Path | None = None,
) -> dict[str, Any]:
    geometry_dir = geometry_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_layers = _available_layers(geometry_dir) if layers is None else layers
    intervention_rows = [] if intervention_root is None else _load_intervention_rows(intervention_root)
    panel_rows = _read_jsonl(geometry_dir / "panel.jsonl") if (geometry_dir / "panel.jsonl").exists() else None
    layer_summaries: list[dict[str, Any]] = []

    for layer in resolved_layers:
        layer_dir = geometry_dir / f"layer_{layer:02d}"
        encoded_path = layer_dir / "encoded_features.npy"
        encoded_features = np.load(encoded_path).astype(np.float32) if encoded_path.exists() else None
        direction_rows = {
            family: _load_direction(np=np, geometry_dir=geometry_dir, layer=layer, family=family)
            for family in CORE_COLOR_FAMILIES
        }
        families = list(direction_rows.keys())
        direction_matrix = np.stack([direction_rows[family] for family in families]).astype(np.float32)
        cosine_matrix = _cosine_similarity_matrix(direction_matrix)
        effective_dimension = _effective_dimensionality(direction_matrix)
        pca_coords, pca_variance = _pca_2d(direction_matrix)
        warm_cool_path = layer_dir / "directions" / "warm_cool_direction.npy"
        warm_cool_direction = np.load(warm_cool_path).astype(np.float32) if warm_cool_path.exists() else None
        warm_cool_projection = _warm_cool_projection(direction_rows, warm_cool_direction)
        opponent = _opponent_summary(families, cosine_matrix)
        feature_attribution = _attribute_directions(
            layer_dir,
            top_k=top_k,
            encoded_features=encoded_features,
            panel_rows=panel_rows,
        )
        intervention_correlation = None
        if (layer_dir / "mean_direction_summary.json").exists():
            intervention_correlation = _intervention_correlation(
                layer=layer,
                direction_summary=_read_json(layer_dir / "mean_direction_summary.json"),
                intervention_rows=intervention_rows,
            )

        report_dir = output_dir / f"layer_{layer:02d}"
        report_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            report_dir / "direction_cosine_similarity.json",
            {
                "families": families,
                "matrix": cosine_matrix.tolist(),
            },
        )
        _write_cosine_heatmap(
            report_dir / "direction_cosine_similarity.svg",
            families=families,
            matrix=cosine_matrix,
            title=f"Layer {layer} direction cosine similarity",
        )
        _write_direction_pca_svg(
            report_dir / "direction_color_wheel.svg",
            families=families,
            coords=pca_coords,
            pc_variance=pca_variance,
            title=f"Layer {layer} direction PCA",
        )
        _write_json(report_dir / "feature_attribution.json", feature_attribution)
        _write_direction_feature_cards(
            report_dir / "direction_feature_cards.md",
            feature_attribution=feature_attribution,
        )
        _write_json(report_dir / "effective_dimensionality.json", effective_dimension)
        _write_json(report_dir / "warm_cool_projection.json", warm_cool_projection)
        _write_json(report_dir / "opponent_structure.json", opponent)
        if intervention_correlation is not None:
            _write_json(report_dir / "intervention_correlation.json", intervention_correlation)

        layer_summary = {
            "components_for_90_percent": effective_dimension["components_for_90_percent"],
            "feature_exemplar_count": sum(
                len(feature_row.get("top_prompts", []))
                for rows in feature_attribution.values()
                for feature_row in rows
            ),
            "intervention_correlation": None if intervention_correlation is None else intervention_correlation["correlation"],
            "layer": layer,
            "top_feature_family_count": sum(1 for rows in feature_attribution.values() if rows),
            "warm_cool_projection": warm_cool_projection,
        }
        _write_json(report_dir / "summary.json", layer_summary)
        layer_summaries.append(layer_summary)

    _write_json(output_dir / "summary.json", {"layers": list(resolved_layers)})
    lines = [
        "# Direction characterization",
        "",
        f"- Geometry dir: `{geometry_dir}`",
        f"- Layers: `{list(resolved_layers)}`",
        "",
        "Per-layer artifacts:",
        "",
        "- `direction_cosine_similarity.json` / `.svg`",
        "- `direction_color_wheel.svg`",
        "- `feature_attribution.json`",
        "- `direction_feature_cards.md`",
        "- `effective_dimensionality.json`",
        "- `warm_cool_projection.json`",
        "- `opponent_structure.json`",
    ]
    if intervention_root is not None:
        lines.extend(["- `intervention_correlation.json`", ""])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"layers": list(resolved_layers)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Characterize geometry directions from a focused SAE run.")
    parser.add_argument("--geometry-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--layers")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--intervention-root", type=Path)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_characterization(
        geometry_dir=args.geometry_dir,
        output_dir=args.output_dir,
        layers=_parse_layers(args.layers),
        top_k=args.top_k,
        intervention_root=args.intervention_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
