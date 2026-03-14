# ruff: noqa: E402

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_script_module(name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_direction_experiment_suite_builds_large_catalog_prompt_files(tmp_path: Path) -> None:
    suite = _load_script_module("direction_experiment_suite", "scripts/direction_experiment_suite.py")
    geometry_dir = tmp_path / "geometry"
    directions_dir = geometry_dir / "layer_11" / "directions"
    directions_dir.mkdir(parents=True, exist_ok=True)
    np.save(directions_dir / "red_direction.npy", np.array([1.0, 0.0], dtype=np.float32))
    np.save(directions_dir / "warm_cool_direction.npy", np.array([0.0, 1.0], dtype=np.float32))

    manifest = suite.build_suite(
        geometry_dir=geometry_dir,
        output_dir=tmp_path / "suite",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        layers=(11,),
        catalog_limit=20,
        alpha_values="-2,0,2",
        max_length=64,
        max_new_tokens=12,
        batch_size=4,
        resume=False,
        device="cpu",
        run_commands=False,
    )

    prompt_root = tmp_path / "suite" / "prompt_files"
    hex_lines = (prompt_root / "describe_catalog_hex_general.txt").read_text(encoding="utf-8").splitlines()
    rgb_lines = (prompt_root / "catalog_word_to_rgb.txt").read_text(encoding="utf-8").splitlines()
    generation_lines = (prompt_root / "catalog_word_to_hex.txt").read_text(encoding="utf-8").splitlines()
    word_lines = (prompt_root / "catalog_hex_to_word.txt").read_text(encoding="utf-8").splitlines()
    assert len(hex_lines) == 20
    assert len(rgb_lines) == 20
    assert len(generation_lines) == 20
    assert len(word_lines) == 20
    assert "exactly three adjectives separated by commas" in hex_lines[0]
    assert "Reply only with a value like 255,0,0." in rgb_lines[0]
    assert "Reply only with a value like #ff0000." in generation_lines[0]
    assert "Return only one color word" in word_lines[0]
    assert manifest["catalog_entry_count"] == 20
    assert manifest["suite_profile"] == "focused"
    assert len(manifest["prompt_sets"]) == 8
    assert manifest["run_count"] == len(manifest["prompt_sets"]) * 2
    assert any("--alpha-values=-2,0,2" in row["command"] for row in manifest["runs"])
    assert (tmp_path / "suite" / "run_suite.sh").exists()


def test_description_space_report_writes_probe_summary(tmp_path: Path) -> None:
    report = _load_script_module("description_space_report", "scripts/description_space_report.py")
    intervention_root = tmp_path / "suite"
    prompt_set_name = "describe_catalog_hex_general"
    suite_manifest = {
        "runs": [],
    }
    descriptor_bank = {
        "red": {
            -1.0: ["muted warm soft", "muted earthy warm"],
            0.0: ["warm vivid glossy", "warm vivid bright"],
            1.0: ["glossy vivid neon", "bright glossy neon"],
        },
        "blue": {
            -1.0: ["muted cool soft", "muted smoky cool"],
            0.0: ["cool vivid glossy", "cool vivid bright"],
            1.0: ["icy glossy neon", "cool glossy neon"],
        },
    }
    for direction_name, alpha_rows in descriptor_bank.items():
        run_dir = intervention_root / "runs" / "layer_11" / direction_name / prompt_set_name
        run_dir.mkdir(parents=True, exist_ok=True)
        suite_manifest["runs"].append(
            {
                "direction_name": direction_name,
                "layer": 11,
                "output_dir": str(run_dir),
                "output_format": "description",
                "prompt_set_name": prompt_set_name,
            }
        )
        rows = []
        for alpha, texts in alpha_rows.items():
            for index, text in enumerate(texts):
                rows.append(
                    {
                        "alpha": alpha,
                        "patched_raw_completion": text,
                        "prompt_id": f"{direction_name}-{alpha}-{index}",
                    }
                )
        (run_dir / "intervention_rows.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )

    manifest_path = tmp_path / "suite_manifest.json"
    manifest_path.write_text(json.dumps(suite_manifest), encoding="utf-8")

    summary = report.run_description_report(
        intervention_root=intervention_root,
        output_dir=tmp_path / "description_report",
        suite_manifest=manifest_path,
    )

    assert summary["total_rows"] == 12
    assert (tmp_path / "description_report" / "probe_summary.json").exists()
    probe_summary = json.loads((tmp_path / "description_report" / "probe_summary.json").read_text(encoding="utf-8"))
    assert probe_summary["rows"][0]["direction_probe_accuracy"] is not None
    assert probe_summary["rows"][0]["alpha_sign_probe_accuracy"] is not None
