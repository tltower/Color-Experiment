# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from color_latent_lab import custom_sae
import color_latent_lab.sae_geometry as sae_geometry


class FakeGeometryTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    unk_token = "<unk>"
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "left"

    def __init__(self) -> None:
        self._next_prompt_id = 10
        self.prompt_meta: dict[int, dict[str, str | None]] = {}
        self._decode_map = {
            100: "#ff0000",
            101: "#0000ff",
            102: "#00ff00",
            103: "red",
            104: "blue",
            105: "green",
        }

    def _parse(self, text: str) -> dict[str, str | None]:
        lowered = text.lower()
        if lowered.startswith("color:"):
            value = text.split(":", 1)[1].strip()
            if value.startswith("#"):
                schema = "hex"
            elif "," in value:
                schema = "rgb"
            else:
                schema = "word"
            family = {
                "red": "red",
                "scarlet": "red",
                "#ff0000": "red",
                "255,0,0": "red",
                "blue": "blue",
                "#0000ff": "blue",
                "0,0,255": "blue",
                "green": "green",
                "#00ff00": "green",
                "0,255,0": "green",
            }.get(value.lower(), "green")
            return {"family": family, "kind": "color", "schema": schema, "subject": None}
        if lowered.startswith("hex code for:"):
            subject = text.split(":", 1)[1].strip().lower() or None
            family = {"fire": "red", "ocean": "blue", "forest": "green"}.get(subject, "green")
            return {"family": family, "kind": "semantic" if subject else "blank", "schema": "hex", "subject": subject}
        if "what color do you associate with the word" in lowered:
            subject = lowered.split("word ", 1)[1].split("?", 1)[0].strip()
            family = {"fire": "red", "ocean": "blue", "forest": "green"}.get(subject, "green")
            schema = "hex" if "hex code" in lowered else "word"
            return {"family": family, "kind": "semantic", "schema": schema, "subject": subject}
        return {"family": "green", "kind": "unknown", "schema": "word", "subject": None}

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert padding is True
        assert truncation is True
        assert return_tensors == "pt"
        rows = []
        for text in texts:
            prompt_id = self._next_prompt_id
            self._next_prompt_id += 1
            self.prompt_meta[prompt_id] = self._parse(text)
            rows.append([2, 3, prompt_id])
        return {
            "input_ids": torch.tensor(rows, dtype=torch.int64),
            "attention_mask": torch.ones((len(rows), 3), dtype=torch.int64),
        }

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return "".join(self._decode_map.get(int(token_id), f"<tok:{int(token_id)}>") for token_id in token_ids)

    def batch_decode(self, rows: torch.Tensor, *, skip_special_tokens: bool) -> list[str]:
        assert skip_special_tokens is True
        return [self._decode_map[int(row[0])] for row in rows.tolist()]


class FakeGeometryBlock(torch.nn.Module):
    def __init__(self, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        updated = hidden_states.clone()
        updated[:, :, 0] = updated[:, :, 0] * (1.1 + 0.15 * self.layer_index)
        updated[:, :, 1] = updated[:, :, 1] * (1.05 + 0.10 * self.layer_index)
        updated[:, :, 2] = updated[:, :, 2] + (self.layer_index + 1) * 0.1
        return updated


class FakeGeometryInner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeGeometryBlock(0), FakeGeometryBlock(1)])


class FakeGeometryModel(torch.nn.Module):
    config = type("Config", (), {"num_hidden_layers": 2, "hidden_size": 5})()

    def __init__(self, tokenizer: FakeGeometryTokenizer) -> None:
        super().__init__()
        self.model = FakeGeometryInner()
        self.tokenizer = tokenizer

    def _family_vector(self, family: str, device: torch.device) -> torch.Tensor:
        if family == "red":
            return torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0], device=device)
        if family == "blue":
            return torch.tensor([-2.0, 0.0, 0.0, 0.0, 0.0], device=device)
        return torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0], device=device)

    def _schema_bias(self, schema: str, device: torch.device) -> torch.Tensor:
        if schema == "hex":
            return torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], device=device)
        if schema == "rgb":
            return torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0], device=device)
        return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=device)

    def forward(self, **kwargs: torch.Tensor):
        input_ids = kwargs["input_ids"]
        device = input_ids.device
        batch_size, sequence_length = input_ids.shape
        hidden = torch.zeros((batch_size, sequence_length, 5), dtype=torch.float32, device=device)
        hidden[:, :, 4] = input_ids.float() / 100.0
        for row_index, prompt_id in enumerate(input_ids[:, -1].tolist()):
            meta = self.tokenizer.prompt_meta[int(prompt_id)]
            hidden[row_index, -1, :] = self._family_vector(str(meta["family"]), device) + self._schema_bias(
                str(meta["schema"]),
                device,
            )
        hidden_states = [hidden.clone()]
        for block in self.model.layers:
            hidden = block(hidden)
            hidden_states.append(hidden.clone())
        return type("Output", (), {"hidden_states": tuple(hidden_states)})()

    def generate(self, **kwargs: torch.Tensor) -> torch.Tensor:
        outputs = self(**kwargs)
        final_hidden = outputs.hidden_states[-1][:, -1, :]
        tokens = []
        for row in final_hidden.tolist():
            if row[0] > 0.8:
                token_id = 100
            elif row[0] < -0.8:
                token_id = 101
            else:
                token_id = 102
            tokens.append([token_id])
        return torch.cat(
            [
                kwargs["input_ids"],
                torch.tensor(tokens, dtype=torch.int64, device=kwargs["input_ids"].device),
            ],
            dim=1,
        )


def _install_fake_geometry_components(monkeypatch) -> None:
    tokenizer = FakeGeometryTokenizer()
    model = FakeGeometryModel(tokenizer)
    monkeypatch.setattr(sae_geometry, "create_generation_components", lambda _model_name: (tokenizer, model))


def _write_fake_sae_repo(root: Path, *, layers: tuple[int, ...]) -> None:
    for layer in layers:
        layer_dir = root / f"resid_post_layer_{layer}" / "trainer_0"
        layer_dir.mkdir(parents=True, exist_ok=True)
        model = custom_sae.SparseAutoencoder(input_dim=5, dictionary_size=4, top_k=None)
        with torch.no_grad():
            model.input_bias.zero_()
            model.encoder.weight.zero_()
            model.encoder.bias.zero_()
            model.decoder.weight.zero_()
            model.encoder.weight[0, 0] = 1.0
            model.encoder.weight[1, 0] = -1.0
            model.encoder.weight[2, 1] = 1.0
            model.encoder.weight[3, 2] = 1.0
            model.decoder.weight[0, 0] = 1.0
            model.decoder.weight[0, 1] = -1.0
            model.decoder.weight[1, 2] = 1.0
            model.decoder.weight[2, 3] = 1.0
        torch.save(
            {
                "config": {"dictionary_size": 4, "input_dim": 5, "top_k": None},
                "state_dict": model.state_dict(),
            },
            layer_dir / "ae.pt",
        )
        (layer_dir / "config.json").write_text(
            json.dumps({"dictionary_size": 4, "input_dim": 5, "top_k": None}),
            encoding="utf-8",
        )


def test_load_off_the_shelf_sae_from_local_checkpoint_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "fake_sae_repo"
    _write_fake_sae_repo(repo_root, layers=(0,))

    model, config = sae_geometry.load_off_the_shelf_sae(
        layer=0,
        repo_id_or_path=str(repo_root),
        trainer_index=0,
        device="cpu",
    )

    assert model.input_dim == 5
    assert model.dictionary_size == 4
    assert config["layer"] == 0
    assert config["repo_id_or_path"] == str(repo_root)


def test_run_color_sae_geometry_experiment_writes_layer_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_geometry_components(monkeypatch)
    monkeypatch.setattr(sae_geometry, "CORE_COLOR_FAMILIES", ("red", "blue", "green"))
    monkeypatch.setattr(
        sae_geometry,
        "CORE_COLOR_HEX",
        {"red": "#ff0000", "blue": "#0000ff", "green": "#00ff00"},
    )
    monkeypatch.setattr(
        sae_geometry,
        "CORE_COLOR_RGB",
        {"red": "255,0,0", "blue": "0,0,255", "green": "0,255,0"},
    )
    repo_root = tmp_path / "fake_sae_repo"
    _write_fake_sae_repo(repo_root, layers=(0, 1))
    word_list_path = tmp_path / "colors.txt"
    word_list_path.write_text("red\nscarlet\nblue\ngreen\n", encoding="utf-8")

    summary = sae_geometry.run_color_sae_geometry_experiment(
        output_dir=tmp_path / "geometry",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        sae_repo_id_or_path=str(repo_root),
        sae_layers=(0, 1),
        trainer_index=0,
        word_list_path=word_list_path,
        batch_size=4,
        encode_batch_size=4,
        max_length=32,
        device="cpu",
    )

    geometry_dir = tmp_path / "geometry"
    assert summary["layers"] == [0, 1]
    assert summary["record_count"] == 13
    assert (geometry_dir / "panel.jsonl").exists()
    assert (geometry_dir / "layer_summary.jsonl").exists()
    assert (geometry_dir / "summary.json").exists()
    assert (geometry_dir / "final_results.json").exists()
    assert (geometry_dir / "layer_00" / "encoded_pca.svg").exists()
    assert (geometry_dir / "layer_00" / "feature_scores.jsonl").exists()
    assert (geometry_dir / "layer_00" / "directions" / "red_direction.npy").exists()
    status = json.loads((geometry_dir / "heartbeat_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "completed"
    checkpoint_state = json.loads((geometry_dir / "checkpoints" / "sae_geometry_state.json").read_text(encoding="utf-8"))
    assert checkpoint_state["capture_complete"] is True
    assert checkpoint_state["completed_layers"] == [0, 1]


def test_run_color_sae_geometry_uses_torch_pca_lowrank(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_geometry_components(monkeypatch)
    monkeypatch.setattr(sae_geometry, "CORE_COLOR_FAMILIES", ("red", "blue", "green"))
    monkeypatch.setattr(
        sae_geometry,
        "CORE_COLOR_HEX",
        {"red": "#ff0000", "blue": "#0000ff", "green": "#00ff00"},
    )
    monkeypatch.setattr(
        sae_geometry,
        "CORE_COLOR_RGB",
        {"red": "255,0,0", "blue": "0,0,255", "green": "0,255,0"},
    )
    repo_root = tmp_path / "fake_sae_repo"
    _write_fake_sae_repo(repo_root, layers=(0,))
    word_list_path = tmp_path / "colors.txt"
    word_list_path.write_text("red\nscarlet\nblue\ngreen\n", encoding="utf-8")

    original_pca_lowrank = torch.pca_lowrank
    pca_call_count = {"count": 0}

    def recording_pca_lowrank(*args, **kwargs):
        pca_call_count["count"] += 1
        return original_pca_lowrank(*args, **kwargs)

    monkeypatch.setattr(torch, "pca_lowrank", recording_pca_lowrank)

    sae_geometry.run_color_sae_geometry_experiment(
        output_dir=tmp_path / "geometry",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        sae_repo_id_or_path=str(repo_root),
        sae_layers=(0,),
        trainer_index=0,
        word_list_path=word_list_path,
        batch_size=4,
        encode_batch_size=4,
        max_length=32,
        device="cpu",
    )

    assert pca_call_count["count"] >= 1


def test_run_color_sae_geometry_can_skip_silhouette(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_geometry_components(monkeypatch)
    monkeypatch.setattr(sae_geometry, "CORE_COLOR_FAMILIES", ("red", "blue", "green"))
    monkeypatch.setattr(
        sae_geometry,
        "CORE_COLOR_HEX",
        {"red": "#ff0000", "blue": "#0000ff", "green": "#00ff00"},
    )
    monkeypatch.setattr(
        sae_geometry,
        "CORE_COLOR_RGB",
        {"red": "255,0,0", "blue": "0,0,255", "green": "0,255,0"},
    )
    repo_root = tmp_path / "fake_sae_repo"
    _write_fake_sae_repo(repo_root, layers=(0,))
    word_list_path = tmp_path / "colors.txt"
    word_list_path.write_text("red\nscarlet\nblue\ngreen\n", encoding="utf-8")
    monkeypatch.setattr(
        sae_geometry,
        "_silhouette_or_none",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("silhouette should be skipped")),
    )

    summary = sae_geometry.run_color_sae_geometry_experiment(
        output_dir=tmp_path / "geometry",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        sae_repo_id_or_path=str(repo_root),
        sae_layers=(0,),
        trainer_index=0,
        word_list_path=word_list_path,
        batch_size=4,
        encode_batch_size=4,
        max_length=32,
        device="cpu",
        compute_silhouette=False,
    )

    assert summary["layers"] == [0]
    layer_summary = json.loads((tmp_path / "geometry" / "layer_00" / "summary.json").read_text(encoding="utf-8"))
    assert layer_summary["family_silhouette"] is None
    assert layer_summary["format_silhouette"] is None


def test_run_color_sae_geometry_uses_raw_prompt_without_chat_wrapper(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_geometry_components(monkeypatch)
    monkeypatch.setattr(sae_geometry, "CORE_COLOR_FAMILIES", ("red", "blue", "green"))
    monkeypatch.setattr(
        sae_geometry,
        "CORE_COLOR_HEX",
        {"red": "#ff0000", "blue": "#0000ff", "green": "#00ff00"},
    )
    monkeypatch.setattr(
        sae_geometry,
        "CORE_COLOR_RGB",
        {"red": "255,0,0", "blue": "0,0,255", "green": "0,255,0"},
    )
    monkeypatch.setattr(
        sae_geometry,
        "_render_prompt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("geometry should use raw prompts")),
    )
    repo_root = tmp_path / "fake_sae_repo"
    _write_fake_sae_repo(repo_root, layers=(0,))
    word_list_path = tmp_path / "colors.txt"
    word_list_path.write_text("red\nscarlet\nblue\ngreen\n", encoding="utf-8")

    sae_geometry.run_color_sae_geometry_experiment(
        output_dir=tmp_path / "geometry",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        sae_repo_id_or_path=str(repo_root),
        sae_layers=(0,),
        trainer_index=0,
        word_list_path=word_list_path,
        batch_size=4,
        encode_batch_size=4,
        max_length=32,
        device="cpu",
    )


def test_run_color_sae_geometry_resume_skips_completed_capture_and_layers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_geometry_components(monkeypatch)
    monkeypatch.setattr(sae_geometry, "CORE_COLOR_FAMILIES", ("red", "blue", "green"))
    monkeypatch.setattr(
        sae_geometry,
        "CORE_COLOR_HEX",
        {"red": "#ff0000", "blue": "#0000ff", "green": "#00ff00"},
    )
    monkeypatch.setattr(
        sae_geometry,
        "CORE_COLOR_RGB",
        {"red": "255,0,0", "blue": "0,0,255", "green": "0,255,0"},
    )
    repo_root = tmp_path / "fake_sae_repo"
    _write_fake_sae_repo(repo_root, layers=(0, 1))
    word_list_path = tmp_path / "colors.txt"
    word_list_path.write_text("red\nscarlet\nblue\ngreen\n", encoding="utf-8")
    geometry_dir = tmp_path / "geometry"

    original_analyze_layer = sae_geometry._analyze_layer
    failure_state = {"raised": False}

    def flaky_analyze_layer(*args, **kwargs):
        if int(kwargs["layer"]) == 1 and not failure_state["raised"]:
            failure_state["raised"] = True
            raise RuntimeError("forced layer-analysis interruption")
        return original_analyze_layer(*args, **kwargs)

    monkeypatch.setattr(sae_geometry, "_analyze_layer", flaky_analyze_layer)

    with pytest.raises(RuntimeError, match="forced layer-analysis interruption"):
        sae_geometry.run_color_sae_geometry_experiment(
            output_dir=geometry_dir,
            model_name="Qwen/Qwen2.5-7B-Instruct",
            sae_repo_id_or_path=str(repo_root),
            sae_layers=(0, 1),
            trainer_index=0,
            word_list_path=word_list_path,
            batch_size=2,
            encode_batch_size=4,
            max_length=32,
            device="cpu",
        )

    assert (geometry_dir / "panel.jsonl").exists()
    assert (geometry_dir / "activations" / "layer_00.npy").exists()
    assert (geometry_dir / "layer_00" / "summary.json").exists()

    monkeypatch.setattr(
        sae_geometry,
        "_capture_last_token_activations",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("capture should not rerun on resume")),
    )
    monkeypatch.setattr(sae_geometry, "_analyze_layer", original_analyze_layer)

    summary = sae_geometry.run_color_sae_geometry_experiment(
        output_dir=geometry_dir,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        sae_repo_id_or_path=str(repo_root),
        sae_layers=(0, 1),
        trainer_index=0,
        word_list_path=word_list_path,
        batch_size=2,
        encode_batch_size=4,
        max_length=32,
        device="cpu",
        resume=True,
    )

    assert summary["layers"] == [0, 1]
    checkpoint_state = json.loads((geometry_dir / "checkpoints" / "sae_geometry_state.json").read_text(encoding="utf-8"))
    assert checkpoint_state["capture_complete"] is True
    assert checkpoint_state["completed_layers"] == [0, 1]


def test_run_color_direction_intervention_experiment_steers_blank_hex(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_geometry_components(monkeypatch)
    monkeypatch.setattr(sae_geometry, "CORE_COLOR_FAMILIES", ("red", "blue", "green"))
    monkeypatch.setattr(
        sae_geometry,
        "CORE_COLOR_HEX",
        {"red": "#ff0000", "blue": "#0000ff", "green": "#00ff00"},
    )
    monkeypatch.setattr(
        sae_geometry,
        "CORE_COLOR_RGB",
        {"red": "255,0,0", "blue": "0,0,255", "green": "0,255,0"},
    )
    repo_root = tmp_path / "fake_sae_repo"
    _write_fake_sae_repo(repo_root, layers=(0, 1))
    word_list_path = tmp_path / "colors.txt"
    word_list_path.write_text("red\nscarlet\nblue\ngreen\n", encoding="utf-8")

    geometry_dir = tmp_path / "geometry"
    sae_geometry.run_color_sae_geometry_experiment(
        output_dir=geometry_dir,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        sae_repo_id_or_path=str(repo_root),
        sae_layers=(1,),
        trainer_index=0,
        word_list_path=word_list_path,
        batch_size=4,
        encode_batch_size=4,
        max_length=32,
        device="cpu",
    )

    summary = sae_geometry.run_color_direction_intervention_experiment(
        output_dir=tmp_path / "intervene",
        geometry_dir=geometry_dir,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        layer=1,
        family="red",
        alpha_values="0,4",
        prompt_mode="blank_hex",
        batch_size=1,
        max_length=32,
        max_new_tokens=4,
        device="cpu",
    )

    assert summary["best_alpha"] == 4.0
    assert summary["best_target_match_rate"] == 1.0
    rows = [
        json.loads(line)
        for line in (tmp_path / "intervene" / "intervention_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baseline_rows = [row for row in rows if float(row["alpha"]) == 0.0]
    patched_rows = [row for row in rows if float(row["alpha"]) == 4.0]
    assert baseline_rows[0]["patched_family"] == "green"
    assert patched_rows[0]["patched_family"] == "red"
    status = json.loads((tmp_path / "intervene" / "heartbeat_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "completed"


def test_run_color_direction_intervention_resume_uses_batch_checkpoints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_geometry_components(monkeypatch)
    monkeypatch.setattr(sae_geometry, "CORE_COLOR_FAMILIES", ("red", "blue", "green"))
    monkeypatch.setattr(
        sae_geometry,
        "CORE_COLOR_HEX",
        {"red": "#ff0000", "blue": "#0000ff", "green": "#00ff00"},
    )
    monkeypatch.setattr(
        sae_geometry,
        "CORE_COLOR_RGB",
        {"red": "255,0,0", "blue": "0,0,255", "green": "0,255,0"},
    )
    monkeypatch.setattr(sae_geometry, "BUILTIN_SEMANTIC_OBJECTS", ("fire", "ocean", "forest"))
    repo_root = tmp_path / "fake_sae_repo"
    _write_fake_sae_repo(repo_root, layers=(1,))
    word_list_path = tmp_path / "colors.txt"
    word_list_path.write_text("red\nscarlet\nblue\ngreen\n", encoding="utf-8")

    geometry_dir = tmp_path / "geometry"
    sae_geometry.run_color_sae_geometry_experiment(
        output_dir=geometry_dir,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        sae_repo_id_or_path=str(repo_root),
        sae_layers=(1,),
        trainer_index=0,
        word_list_path=word_list_path,
        batch_size=4,
        encode_batch_size=4,
        max_length=32,
        device="cpu",
    )

    tokenizer = FakeGeometryTokenizer()
    model = FakeGeometryModel(tokenizer)
    original_generate = model.generate
    generate_calls = {"count": 0}

    def flaky_generate(**kwargs):
        generate_calls["count"] += 1
        if generate_calls["count"] == 4:
            raise RuntimeError("forced intervention interruption")
        return original_generate(**kwargs)

    monkeypatch.setattr(model, "generate", flaky_generate)
    monkeypatch.setattr(sae_geometry, "create_generation_components", lambda _model_name: (tokenizer, model))

    intervene_dir = tmp_path / "intervene"
    with pytest.raises(RuntimeError, match="forced intervention interruption"):
        sae_geometry.run_color_direction_intervention_experiment(
            output_dir=intervene_dir,
            geometry_dir=geometry_dir,
            model_name="Qwen/Qwen2.5-7B-Instruct",
            layer=1,
            family="red",
            alpha_values="0,4",
            prompt_mode="semantic_hex",
            batch_size=1,
            max_length=32,
            max_new_tokens=4,
            device="cpu",
        )

    assert (intervene_dir / "checkpoints" / "intervention_batches" / "batch_0001.jsonl").exists()

    _install_fake_geometry_components(monkeypatch)
    summary = sae_geometry.run_color_direction_intervention_experiment(
        output_dir=intervene_dir,
        geometry_dir=geometry_dir,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        layer=1,
        family="red",
        alpha_values="0,4",
        prompt_mode="semantic_hex",
        batch_size=1,
        max_length=32,
        max_new_tokens=4,
        device="cpu",
        resume=True,
    )

    assert summary["best_alpha"] == 4.0
    checkpoint_state = json.loads((intervene_dir / "checkpoints" / "sae_intervene_state.json").read_text(encoding="utf-8"))
    assert checkpoint_state["completed_batches"] == [1, 2, 3]
    rows = [
        json.loads(line)
        for line in (intervene_dir / "intervention_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 6
