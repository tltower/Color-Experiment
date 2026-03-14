#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from color_latent_lab.analysis_common import (
    available_direction_layers as _available_layers,
    parse_layers as _parse_layers,
    write_json as _write_json,
)
from color_latent_lab.color_palette import approximate_color_word_hex, hex_to_rgb_string
from color_latent_lab.sae_geometry import (
    BUILTIN_SEMANTIC_OBJECTS,
    CORE_COLOR_FAMILIES,
    CORE_COLOR_HEX,
    CORE_COLOR_RGB,
)
from color_latent_lab.word_lists import bundled_color_word_list_path, read_word_file

STYLE_SEMANTIC_WORDS: tuple[str, ...] = (
    "velvet",
    "chrome",
    "pearl",
    "smoke",
    "rust",
    "linen",
    "moss",
    "ice",
    "sand",
    "ash",
    "wine",
    "platinum",
)


@dataclass(frozen=True)
class PromptSet:
    name: str
    output_format: str
    prompt_mode: str | None = None
    prompt_file_name: str | None = None
    prompt_count: int = 0
    category: str = "description"

def _available_directions(geometry_dir: Path, *, layer: int) -> tuple[str, ...]:
    direction_dir = geometry_dir / f"layer_{layer:02d}" / "directions"
    names: list[str] = []
    for path in sorted(direction_dir.glob("*_direction.npy")):
        names.append(path.stem[: -len("_direction")])
    if not names:
        raise FileNotFoundError(f"No direction files found under {direction_dir}")
    return tuple(names)


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _load_catalog_entries(limit: int | None) -> list[dict[str, str]]:
    words = read_word_file(bundled_color_word_list_path(), limit=limit)
    rows: list[dict[str, str]] = []
    for word in words:
        approximate_hex = approximate_color_word_hex(word)
        if approximate_hex is None:
            continue
        rows.append(
            {
                "word": word,
                "hex": approximate_hex,
                "rgb": hex_to_rgb_string(approximate_hex),
            }
        )
    if not rows:
        raise ValueError("Could not build any approximate catalog color entries.")
    return rows


def _core_entries() -> list[dict[str, str]]:
    return [
        {
            "word": family,
            "hex": CORE_COLOR_HEX[family],
            "rgb": CORE_COLOR_RGB[family],
        }
        for family in CORE_COLOR_FAMILIES
    ]


def _prompt_specs_for_profile(
    *,
    catalog_entries: list[dict[str, str]],
    suite_profile: str,
) -> list[PromptSet]:
    core_entries = _core_entries()
    semantic_words = list(BUILTIN_SEMANTIC_OBJECTS) + [word for word in STYLE_SEMANTIC_WORDS if word not in BUILTIN_SEMANTIC_OBJECTS]

    focused_specs: list[tuple[PromptSet, list[str]]] = [
        (
            PromptSet(name="blank_hex", output_format="hex", prompt_mode="blank_hex", prompt_count=1, category="generation"),
            [],
        ),
        (
            PromptSet(
                name="semantic_hex",
                output_format="hex",
                prompt_mode="semantic_hex",
                prompt_count=len(BUILTIN_SEMANTIC_OBJECTS),
                category="generation",
            ),
            [],
        ),
        (
            PromptSet(
                name="catalog_word_to_hex",
                output_format="hex",
                prompt_file_name="catalog_word_to_hex.txt",
                prompt_count=len(catalog_entries),
                category="generation",
            ),
            [f"Hex code for: {row['word']}" for row in catalog_entries],
        ),
        (
            PromptSet(
                name="catalog_word_to_rgb",
                output_format="rgb",
                prompt_file_name="catalog_word_to_rgb.txt",
                prompt_count=len(catalog_entries),
                category="generation",
            ),
            [f"RGB triplet for: {row['word']}" for row in catalog_entries],
        ),
        (
            PromptSet(
                name="catalog_hex_to_word",
                output_format="word",
                prompt_file_name="catalog_hex_to_word.txt",
                prompt_count=len(catalog_entries),
                category="generation",
            ),
            [f"Color word for: {row['hex']}" for row in catalog_entries],
        ),
        (
            PromptSet(
                name="describe_core_word_general",
                output_format="description",
                prompt_file_name="describe_core_word_general.txt",
                prompt_count=len(core_entries),
            ),
            [
                f"Describe the appearance of the color {row['word']} in three adjectives. Do not name objects."
                for row in core_entries
            ],
        ),
        (
            PromptSet(
                name="describe_core_hex_general",
                output_format="description",
                prompt_file_name="describe_core_hex_general.txt",
                prompt_count=len(core_entries),
            ),
            [
                f"Describe the appearance of the color {row['hex']} in three adjectives. Do not name objects."
                for row in core_entries
            ],
        ),
        (
            PromptSet(
                name="describe_core_rgb_general",
                output_format="description",
                prompt_file_name="describe_core_rgb_general.txt",
                prompt_count=len(core_entries),
            ),
            [
                f"Describe the appearance of the color {row['rgb']} in three adjectives. Do not name objects."
                for row in core_entries
            ],
        ),
        (
            PromptSet(
                name="describe_catalog_word_general",
                output_format="description",
                prompt_file_name="describe_catalog_word_general.txt",
                prompt_count=len(catalog_entries),
            ),
            [
                f"Describe the appearance of the color {row['word']} in three adjectives. Do not name objects."
                for row in catalog_entries
            ],
        ),
        (
            PromptSet(
                name="describe_catalog_hex_general",
                output_format="description",
                prompt_file_name="describe_catalog_hex_general.txt",
                prompt_count=len(catalog_entries),
            ),
            [
                f"Describe the appearance of the color {row['hex']} in three adjectives. Do not name objects."
                for row in catalog_entries
            ],
        ),
        (
            PromptSet(
                name="describe_catalog_rgb_general",
                output_format="description",
                prompt_file_name="describe_catalog_rgb_general.txt",
                prompt_count=len(catalog_entries),
            ),
            [
                f"Describe the appearance of the color {row['rgb']} in three adjectives. Do not name objects."
                for row in catalog_entries
            ],
        ),
        (
            PromptSet(
                name="describe_catalog_hex_style",
                output_format="description",
                prompt_file_name="describe_catalog_hex_style.txt",
                prompt_count=len(catalog_entries),
            ),
            [
                f"Describe the color {row['hex']} in terms of finish, vividness, and mood using three adjectives."
                for row in catalog_entries
            ],
        ),
        (
            PromptSet(
                name="describe_semantic_words",
                output_format="description",
                prompt_file_name="describe_semantic_words.txt",
                prompt_count=len(semantic_words),
            ),
            [
                f"Describe the color of the following word in three adjectives: {word}"
                for word in semantic_words
            ],
        ),
    ]
    if suite_profile == "focused":
        prompt_specs = [
            prompt_spec
            for prompt_spec in focused_specs
            if prompt_spec[0].name
            in {
                "blank_hex",
                "semantic_hex",
                "catalog_word_to_hex",
                "catalog_word_to_rgb",
                "catalog_hex_to_word",
                "describe_catalog_hex_general",
                "describe_catalog_hex_style",
                "describe_semantic_words",
            }
        ]
    elif suite_profile == "extended":
        prompt_specs = focused_specs
    else:
        raise ValueError(f"Unsupported suite profile: {suite_profile}")
    return prompt_specs


def _write_prompt_files(
    *,
    output_dir: Path,
    catalog_entries: list[dict[str, str]],
    suite_profile: str,
) -> list[PromptSet]:
    prompt_dir = output_dir / "prompt_files"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_specs = _prompt_specs_for_profile(
        catalog_entries=catalog_entries,
        suite_profile=suite_profile,
    )

    written: list[PromptSet] = []
    for prompt_set, lines in prompt_specs:
        if prompt_set.prompt_file_name is not None:
            _write_lines(prompt_dir / prompt_set.prompt_file_name, lines)
        written.append(prompt_set)
    return written


def _command_for_entry(
    *,
    repo_root: Path,
    prompt_root: Path,
    model_name: str,
    geometry_dir: Path,
    run_output_dir: Path,
    layer: int,
    direction_name: str,
    prompt_set: PromptSet,
    alpha_values: str,
    max_length: int,
    max_new_tokens: int,
    batch_size: int,
    resume: bool,
    device: str,
) -> list[str]:
    command = [
        str(repo_root / ".venv" / "bin" / "color-latent-lab")
        if (repo_root / ".venv" / "bin" / "color-latent-lab").exists()
        else "color-latent-lab",
        "sae-intervene",
        "--model-name",
        model_name,
        "--geometry-dir",
        str(geometry_dir),
        "--output-dir",
        str(run_output_dir),
        "--layer",
        str(layer),
        "--family",
        direction_name,
        f"--alpha-values={alpha_values}",
        "--output-format",
        prompt_set.output_format,
        "--batch-size",
        str(batch_size),
        "--max-length",
        str(max_length),
        "--max-new-tokens",
        str(max_new_tokens),
        "--device",
        device,
    ]
    if prompt_set.prompt_mode is not None:
        command.extend(["--prompt-mode", prompt_set.prompt_mode])
    if prompt_set.prompt_file_name is not None:
        command.extend(["--prompt-file", str((prompt_root / prompt_set.prompt_file_name).resolve())])
    if resume:
        command.append("--resume")
    return command
def build_suite(
    *,
    geometry_dir: Path,
    output_dir: Path,
    model_name: str,
    layers: tuple[int, ...] | None,
    catalog_limit: int | None,
    suite_profile: str = "focused",
    alpha_values: str,
    max_length: int,
    max_new_tokens: int,
    batch_size: int,
    resume: bool,
    device: str,
    run_commands: bool,
) -> dict[str, Any]:
    geometry_dir = geometry_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_layers = _available_layers(geometry_dir) if layers is None else layers
    catalog_entries = _load_catalog_entries(catalog_limit)
    prompt_sets = _write_prompt_files(
        output_dir=output_dir,
        catalog_entries=catalog_entries,
        suite_profile=suite_profile,
    )
    prompt_root = output_dir / "prompt_files"
    manifest_rows: list[dict[str, Any]] = []
    shell_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]

    for layer in resolved_layers:
        direction_names = _available_directions(geometry_dir, layer=layer)
        for direction_name in direction_names:
            for prompt_set in prompt_sets:
                run_output_dir = output_dir / "runs" / f"layer_{layer:02d}" / direction_name / prompt_set.name
                command = _command_for_entry(
                    repo_root=REPO_ROOT,
                    prompt_root=prompt_root,
                    model_name=model_name,
                    geometry_dir=geometry_dir,
                    run_output_dir=run_output_dir,
                    layer=layer,
                    direction_name=direction_name,
                    prompt_set=prompt_set,
                    alpha_values=alpha_values,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                    batch_size=batch_size,
                    resume=resume,
                    device=device,
                )
                manifest_row = {
                    "alpha_values": alpha_values,
                    "batch_size": batch_size,
                    "category": prompt_set.category,
                    "command": command,
                    "direction_name": direction_name,
                    "geometry_dir": str(geometry_dir),
                    "layer": layer,
                    "output_dir": str(run_output_dir),
                    "output_format": prompt_set.output_format,
                    "prompt_count": prompt_set.prompt_count,
                    "prompt_file": None
                    if prompt_set.prompt_file_name is None
                    else str((output_dir / "prompt_files" / prompt_set.prompt_file_name).resolve()),
                    "prompt_mode": prompt_set.prompt_mode,
                    "prompt_set_name": prompt_set.name,
                }
                manifest_rows.append(manifest_row)
                shell_lines.append(shlex.join(command))

    manifest = {
        "catalog_entry_count": len(catalog_entries),
        "layers": list(resolved_layers),
        "model_name": model_name,
        "prompt_sets": [asdict(prompt_set) for prompt_set in prompt_sets],
        "run_count": len(manifest_rows),
        "suite_profile": suite_profile,
        "runs": manifest_rows,
    }
    _write_json(output_dir / "suite_manifest.json", manifest)
    _write_lines(output_dir / "run_suite.sh", shell_lines)
    (output_dir / "run_suite.sh").chmod(0o755)
    report_lines = [
        "# Direction experiment suite",
        "",
        f"- Geometry dir: `{geometry_dir}`",
        f"- Layers: `{list(resolved_layers)}`",
        f"- Catalog entry count: `{len(catalog_entries)}`",
        f"- Prompt-set count: `{len(prompt_sets)}`",
        f"- Suite profile: `{suite_profile}`",
        f"- Run count: `{len(manifest_rows)}`",
        "",
        "Artifacts:",
        "",
        "- `suite_manifest.json`",
        "- `run_suite.sh`",
        "- `prompt_files/`",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    results: list[dict[str, Any]] = []
    if run_commands:
        for row in manifest_rows:
            completed = subprocess.run(row["command"], cwd=str(REPO_ROOT), check=False)
            results.append(
                {
                    "direction_name": row["direction_name"],
                    "layer": row["layer"],
                    "output_dir": row["output_dir"],
                    "prompt_set_name": row["prompt_set_name"],
                    "returncode": completed.returncode,
                }
            )
        _write_json(output_dir / "run_results.json", {"results": results})
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and optionally run a broad direction experiment suite.")
    parser.add_argument("--geometry-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layers")
    parser.add_argument("--catalog-limit", type=int, default=96)
    parser.add_argument("--suite-profile", default="focused", choices=("focused", "extended"))
    parser.add_argument("--alpha-values", default="-8,-4,-2,-1,0,1,2,4,8")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    build_suite(
        geometry_dir=args.geometry_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        layers=_parse_layers(args.layers),
        catalog_limit=args.catalog_limit,
        suite_profile=args.suite_profile,
        alpha_values=args.alpha_values,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        resume=args.resume,
        device=args.device,
        run_commands=args.run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
