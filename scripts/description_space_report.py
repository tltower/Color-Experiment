#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import html
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from color_latent_lab.analysis_common import (
    cosine_similarity_matrix as _cosine_similarity_matrix,
    read_json as _read_json,
    read_jsonl as _read_jsonl,
    write_json as _write_json,
)

TOKEN_RE = re.compile(r"[a-z]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "color",
    "describe",
    "following",
    "for",
    "in",
    "is",
    "it",
    "its",
    "of",
    "or",
    "the",
    "this",
    "three",
    "to",
    "using",
    "word",
}
DESCRIPTOR_SYNONYMS = {
    "bright": "vivid",
    "brilliant": "vivid",
    "electric": "vivid",
    "vibrant": "vivid",
    "pale": "light",
    "soft": "muted",
    "subtle": "muted",
    "dusty": "muted",
    "washed": "muted",
    "deep": "dark",
    "inky": "dark",
    "shiny": "glossy",
    "lustrous": "glossy",
    "gleaming": "glossy",
    "metal": "metallic",
    "chromed": "metallic",
    "pastel": "pastel",
    "neon": "neon",
}


def _require_probe_stack() -> tuple[Any, Any]:
    try:
        from sklearn.linear_model import RidgeClassifier  # type: ignore[import-not-found]
        from sklearn.model_selection import StratifiedKFold  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "The research stack is not installed. Use `pip install -e .[dev]` in the repo environment."
        ) from exc
    return RidgeClassifier, StratifiedKFold
def _normalize_description(text: str) -> list[str]:
    tokens: list[str] = []
    for token in TOKEN_RE.findall(text.lower()):
        if token in STOPWORDS:
            continue
        normalized = DESCRIPTOR_SYNONYMS.get(token, token)
        if normalized in STOPWORDS:
            continue
        tokens.append(normalized)
    return tokens


def _load_suite_rows(suite_manifest: Path | None, intervention_root: Path) -> list[dict[str, Any]]:
    if suite_manifest is not None and suite_manifest.exists():
        manifest = _read_json(suite_manifest)
        return [row for row in manifest.get("runs", []) if row.get("output_format") == "description"]
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(intervention_root.rglob("summary.json")):
        summary = _read_json(summary_path)
        if summary.get("output_format") != "description":
            continue
        rows.append(
            {
                "direction_name": summary.get("direction_name", summary.get("target_family")),
                "layer": summary["layer"],
                "output_dir": str(summary_path.parent),
                "output_format": "description",
                "prompt_set_name": summary.get("prompt_mode", summary_path.parent.name),
            }
        )
    return rows


def _build_vocab(run_rows: list[dict[str, Any]]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for row in run_rows:
        for token in row["tokens"]:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def _vectorize(tokens: list[str], vocab: dict[str, int]) -> np.ndarray:
    vector = np.zeros(len(vocab), dtype=np.float32)
    for token, count in Counter(tokens).items():
        index = vocab.get(token)
        if index is not None:
            vector[index] = float(count)
    return vector
def _write_heatmap(path: Path, *, labels: list[str], matrix: np.ndarray, title: str) -> None:
    cell = 56
    margin = 170
    width = margin + cell * len(labels) + 24
    height = margin + cell * len(labels) + 24

    def color_for(value: float) -> str:
        bounded = max(0.0, min(1.0, value))
        shade = int(245 - bounded * 160)
        return f"#ff{shade:02x}{shade:02x}"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7" />',
        f'<text x="24" y="34" font-family="Helvetica, Arial, sans-serif" font-size="24" fill="#111111">{html.escape(title)}</text>',
    ]
    for index, label in enumerate(labels):
        x = margin + index * cell
        y = margin - 12
        lines.append(
            f'<text x="{x + cell / 2:.1f}" y="{y:.1f}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="10" fill="#222222">{html.escape(label)}</text>'
        )
        lines.append(
            f'<text x="{margin - 12:.1f}" y="{margin + index * cell + cell / 2 + 4:.1f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="10" fill="#222222">{html.escape(label)}</text>'
        )
    for row_index, _ in enumerate(labels):
        for col_index, _ in enumerate(labels):
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


def _stratified_accuracy(
    *,
    values: np.ndarray,
    labels: list[str],
) -> dict[str, Any] | None:
    RidgeClassifier, StratifiedKFold = _require_probe_stack()
    label_counts = Counter(labels)
    if not label_counts or min(label_counts.values()) < 2:
        return None
    split_count = min(5, min(label_counts.values()))
    splitter = StratifiedKFold(n_splits=split_count, shuffle=True, random_state=17)
    label_array = np.array(labels)
    scores: list[float] = []
    for train_indices, test_indices in splitter.split(np.arange(len(labels)).reshape(-1, 1), label_array):
        model = RidgeClassifier(class_weight="balanced")
        model.fit(values[train_indices], label_array[train_indices])
        predicted = model.predict(values[test_indices])
        scores.append(float((predicted == label_array[test_indices]).mean()))
    return {"accuracy": float(sum(scores) / len(scores)), "fold_scores": scores}


def run_description_report(
    *,
    intervention_root: Path,
    output_dir: Path,
    suite_manifest: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_rows = _load_suite_rows(suite_manifest, intervention_root)
    if not suite_rows:
        raise FileNotFoundError("No description-format intervention runs found.")

    collected_rows: list[dict[str, Any]] = []
    for row in suite_rows:
        run_dir = Path(row["output_dir"])
        interventions_path = run_dir / "intervention_rows.jsonl"
        if not interventions_path.exists():
            continue
        for intervention_row in _read_jsonl(interventions_path):
            tokens = _normalize_description(str(intervention_row.get("patched_raw_completion", "")))
            collected_rows.append(
                {
                    "alpha": float(intervention_row["alpha"]),
                    "direction_name": str(row["direction_name"]),
                    "layer": int(row["layer"]),
                    "prompt_set_name": str(row["prompt_set_name"]),
                    "run_dir": str(run_dir),
                    "text": str(intervention_row.get("patched_raw_completion", "")),
                    "tokens": tokens,
                }
            )
    if not collected_rows:
        raise FileNotFoundError("Description runs were listed, but no intervention_rows.jsonl files were available.")

    vocab = _build_vocab(collected_rows)
    if not vocab:
        raise ValueError("Could not extract any normalized descriptor tokens from the description runs.")
    values = np.stack([_vectorize(row["tokens"], vocab) for row in collected_rows]).astype(np.float32)

    group_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(collected_rows):
        grouped[(row["layer"], row["prompt_set_name"], row["direction_name"])].append(index)

    for (layer, prompt_set_name, direction_name), indices in sorted(grouped.items()):
        alpha_groups: dict[float, list[int]] = defaultdict(list)
        for index in indices:
            alpha_groups[float(collected_rows[index]["alpha"])].append(index)
        labels = [str(alpha) for alpha in sorted(alpha_groups)]
        centroid_matrix = np.stack(
            [
                values[np.array(alpha_groups[alpha], dtype=np.int64)].mean(axis=0)
                for alpha in sorted(alpha_groups)
            ]
        ).astype(np.float32)
        cosine = _cosine_similarity_matrix(centroid_matrix)
        safe_name = f"layer_{layer:02d}_{direction_name}_{prompt_set_name}".replace("/", "_")
        _write_json(
            output_dir / f"{safe_name}.alpha_centroids.json",
            {
                "alphas": [float(alpha) for alpha in sorted(alpha_groups)],
                "direction_name": direction_name,
                "layer": layer,
                "prompt_set_name": prompt_set_name,
            },
        )
        _write_heatmap(
            output_dir / f"{safe_name}.alpha_cosine.svg",
            labels=labels,
            matrix=cosine,
            title=f"Layer {layer} {direction_name} {prompt_set_name}",
        )
        baseline_indices = alpha_groups.get(0.0)
        top_shift_tokens: list[dict[str, Any]] = []
        if baseline_indices is not None:
            baseline_mean = values[np.array(baseline_indices, dtype=np.int64)].mean(axis=0)
            for alpha, alpha_indices in sorted(alpha_groups.items()):
                alpha_mean = values[np.array(alpha_indices, dtype=np.int64)].mean(axis=0)
                delta = alpha_mean - baseline_mean
                if not np.any(delta):
                    continue
                top_indices = np.argsort(-np.abs(delta))[:8]
                top_shift_tokens.extend(
                    {
                        "alpha": float(alpha),
                        "delta": float(delta[index]),
                        "token": token,
                    }
                    for token, index in vocab.items()
                    if index in set(int(value) for value in top_indices)
                )
        group_rows.append(
            {
                "alpha_count": len(alpha_groups),
                "direction_name": direction_name,
                "layer": layer,
                "point_count": len(indices),
                "prompt_set_name": prompt_set_name,
                "top_shift_tokens": top_shift_tokens,
            }
        )

    probe_rows: list[dict[str, Any]] = []
    slice_groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, row in enumerate(collected_rows):
        slice_groups[(row["layer"], row["prompt_set_name"])].append(index)
    for (layer, prompt_set_name), indices in sorted(slice_groups.items()):
        slice_values = values[np.array(indices, dtype=np.int64)]
        direction_labels = [collected_rows[index]["direction_name"] for index in indices]
        alpha_sign_labels = [
            "negative" if collected_rows[index]["alpha"] < 0 else "positive" if collected_rows[index]["alpha"] > 0 else "zero"
            for index in indices
        ]
        direction_probe = _stratified_accuracy(values=slice_values, labels=direction_labels)
        alpha_probe = _stratified_accuracy(values=slice_values, labels=alpha_sign_labels)
        probe_rows.append(
            {
                "alpha_sign_probe_accuracy": None if alpha_probe is None else alpha_probe["accuracy"],
                "direction_probe_accuracy": None if direction_probe is None else direction_probe["accuracy"],
                "layer": layer,
                "point_count": len(indices),
                "prompt_set_name": prompt_set_name,
            }
        )

    summary = {
        "description_run_count": len(group_rows),
        "prompt_sets": sorted({row["prompt_set_name"] for row in collected_rows}),
        "token_vocab_size": len(vocab),
        "total_rows": len(collected_rows),
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "group_summary.json", {"rows": group_rows})
    _write_json(output_dir / "probe_summary.json", {"rows": probe_rows})
    lines = [
        "# Description-space report",
        "",
        f"- Total description completions: `{summary['total_rows']}`",
        f"- Description run slices: `{summary['description_run_count']}`",
        f"- Descriptor vocabulary size: `{summary['token_vocab_size']}`",
        "",
        "Primary outputs:",
        "",
        "- `group_summary.json`",
        "- `probe_summary.json`",
        "- `*.alpha_cosine.svg` heatmaps",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze description-mode intervention outputs with shared-word cosine and linear probes.")
    parser.add_argument("--intervention-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--suite-manifest", type=Path)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_description_report(
        intervention_root=args.intervention_root,
        output_dir=args.output_dir,
        suite_manifest=args.suite_manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
