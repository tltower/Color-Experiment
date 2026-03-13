# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import color_latent_lab.logit_lens as logit_lens
import color_latent_lab.custom_sae as custom_sae
import color_latent_lab.word_lists as word_lists
from color_latent_lab import experiment

WORD_FAMILIES = {
    "fire": "red",
    "rose": "red",
    "ocean": "blue",
    "sky": "blue",
    "forest": "green",
    "moss": "green",
}


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    unk_token = "<unk>"
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "right"

    def __init__(self) -> None:
        self.format_to_id = {"word": 11, "hex": 12, "rgb": 13}
        self._word_to_id: dict[str, int] = {}
        self._id_to_word: dict[int, str] = {}
        self._decode_map = {
            100: "red",
            101: "blue",
            102: "green",
            200: "#ff0000",
            201: "#0000ff",
            202: "#00ff00",
            300: "255,0,0",
            301: "0,0,255",
            302: "0,255,0",
        }
        self._literal_to_id = {
            "red": 100,
            " red": 100,
            "blue": 101,
            " blue": 101,
            "green": 102,
            " green": 102,
            "#ff0000": 200,
            " #ff0000": 200,
            "#0000ff": 201,
            " #0000ff": 201,
            "#00ff00": 202,
            " #00ff00": 202,
            "255,0,0": 300,
            " 255,0,0": 300,
            "0,0,255": 301,
            " 0,0,255": 301,
            "0,255,0": 302,
            " 0,255,0": 302,
        }

    def _word_id(self, text: str) -> int:
        word = text.split("word ", 1)[1].split("?", 1)[0].strip().lower()
        if word not in self._word_to_id:
            identifier = len(self._word_to_id) + 20
            self._word_to_id[word] = identifier
            self._id_to_word[identifier] = word
        return self._word_to_id[word]

    def _format_id(self, text: str) -> int:
        lowered = text.lower()
        if "hex code" in lowered:
            return self.format_to_id["hex"]
        if "color word" in lowered:
            return self.format_to_id["word"]
        return self.format_to_id["rgb"]

    def family_for_word_id(self, word_id: int) -> str:
        return WORD_FAMILIES[self._id_to_word[word_id]]

    def output_token_id(self, format_id: int, family: str) -> int:
        if format_id == self.format_to_id["word"]:
            return {"red": 100, "blue": 101, "green": 102}[family]
        if format_id == self.format_to_id["hex"]:
            return {"red": 200, "blue": 201, "green": 202}[family]
        return {"red": 300, "blue": 301, "green": 302}[family]

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
        assert max_length == 96
        assert return_tensors == "pt"
        rows = [[2, self._word_id(text), self._format_id(text)] for text in texts]
        return {
            "input_ids": torch.tensor(rows, dtype=torch.int64),
            "attention_mask": torch.ones((len(rows), 3), dtype=torch.int64),
        }

    def batch_decode(self, rows: torch.Tensor, *, skip_special_tokens: bool) -> list[str]:
        assert skip_special_tokens is True
        decoded: list[str] = []
        for row in rows.tolist():
            decoded.append(self._decode_map[int(row[0])])
        return decoded

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        token_id = self._literal_to_id.get(text)
        if token_id is not None:
            return [token_id]
        token_id = self._literal_to_id.get(text.strip())
        if token_id is not None:
            return [token_id]
        return []

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


class FakeBlock(torch.nn.Module):
    def __init__(self, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        updated = hidden_states.clone()
        updated[:, :, 0] = updated[:, :, 0] * (1.05 + self.layer_index * 0.03)
        updated[:, :, 1] = updated[:, :, 1] * (1.02 + self.layer_index * 0.02)
        updated[:, :, 3] = updated[:, :, 3] + (self.layer_index + 1) * 0.05
        return updated


class FakeInner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeBlock(0), FakeBlock(1)])
        self.norm = torch.nn.Identity()


class FakeModel(torch.nn.Module):
    config = type("Config", (), {"num_hidden_layers": 2, "hidden_size": 4})()

    def __init__(self, tokenizer: FakeTokenizer) -> None:
        super().__init__()
        self.model = FakeInner()
        self.tokenizer = tokenizer
        self.lm_head = torch.nn.Linear(4, 512, bias=False)
        with torch.no_grad():
            self.lm_head.weight.zero_()
            self.lm_head.weight[100] = torch.tensor([4.0, 0.0, 0.0, 0.0])
            self.lm_head.weight[101] = torch.tensor([-4.0, 0.0, 0.0, 0.0])
            self.lm_head.weight[102] = torch.tensor([0.0, 4.0, 0.0, 0.0])
            self.lm_head.weight[200] = torch.tensor([2.0, 0.0, 10.0, 0.0])
            self.lm_head.weight[201] = torch.tensor([-2.0, 0.0, 10.0, 0.0])
            self.lm_head.weight[202] = torch.tensor([0.0, 2.0, 10.0, 0.0])
            self.lm_head.weight[300] = torch.tensor([2.0, 0.0, 12.0, 0.0])
            self.lm_head.weight[301] = torch.tensor([-2.0, 0.0, 12.0, 0.0])
            self.lm_head.weight[302] = torch.tensor([0.0, 2.0, 12.0, 0.0])

    def _family_vector(self, family: str, device: torch.device) -> torch.Tensor:
        if family == "red":
            return torch.tensor([1.0, 0.1, 0.0, 0.0], device=device)
        if family == "blue":
            return torch.tensor([-1.0, 0.1, 0.0, 0.0], device=device)
        return torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)

    def _format_bias(self, format_id: int, device: torch.device) -> torch.Tensor:
        if format_id == self.tokenizer.format_to_id["word"]:
            return torch.tensor([0.00, 0.00, 0.10, 0.0], device=device)
        if format_id == self.tokenizer.format_to_id["hex"]:
            return torch.tensor([0.08, 0.00, 0.30, 0.0], device=device)
        return torch.tensor([-0.08, 0.00, 0.50, 0.0], device=device)

    def forward(self, **kwargs: torch.Tensor):
        input_ids = kwargs["input_ids"]
        device = input_ids.device
        batch_size, sequence_length = input_ids.shape
        hidden = torch.zeros((batch_size, sequence_length, 4), dtype=torch.float32, device=device)
        hidden[:, :, 2] = input_ids.float() / 100.0
        word_ids = input_ids[:, 1].tolist()
        format_ids = input_ids[:, 2].tolist()
        for row_index, (word_id, format_id) in enumerate(zip(word_ids, format_ids, strict=True)):
            family = self.tokenizer.family_for_word_id(int(word_id))
            hidden[row_index, -1, :] = (
                self._family_vector(family, device)
                + self._format_bias(int(format_id), device)
                + torch.tensor([0.0, 0.0, 0.0, float(word_id) / 100.0], device=device)
            )
        hidden_states = [hidden.clone()]
        for block in self.model.layers:
            hidden = block(hidden)
            hidden_states.append(hidden.clone())
        return type("Output", (), {"hidden_states": tuple(hidden_states)})()

    def generate(self, **kwargs: torch.Tensor) -> torch.Tensor:
        outputs = self(**kwargs)
        final_hidden = outputs.hidden_states[-1][:, -1, :]
        format_ids = kwargs["input_ids"][:, 2].tolist()
        new_tokens = []
        for row, format_id in zip(final_hidden.tolist(), format_ids, strict=True):
            if row[0] > 0.55:
                family = "red"
            elif row[0] < -0.55:
                family = "blue"
            else:
                family = "green"
            new_tokens.append([self.tokenizer.output_token_id(int(format_id), family)])
        return torch.cat(
            [
                kwargs["input_ids"],
                torch.tensor(new_tokens, dtype=torch.int64, device=kwargs["input_ids"].device),
            ],
            dim=1,
        )

    def get_output_embeddings(self) -> torch.nn.Linear:
        return self.lm_head


class FailingFakeModel(FakeModel):
    def __init__(self, tokenizer: FakeTokenizer, *, fail_after_generate_calls: int) -> None:
        super().__init__(tokenizer)
        self.fail_after_generate_calls = fail_after_generate_calls
        self.generate_calls = 0

    def generate(self, **kwargs: torch.Tensor) -> torch.Tensor:
        self.generate_calls += 1
        if self.generate_calls > self.fail_after_generate_calls:
            raise RuntimeError("simulated interruption")
        return super().generate(**kwargs)


def _install_fake_components(monkeypatch, *, fail_after_generate_calls: int | None = None) -> None:
    tokenizer = FakeTokenizer()
    if fail_after_generate_calls is None:
        model = FakeModel(tokenizer)
    else:
        model = FailingFakeModel(tokenizer, fail_after_generate_calls=fail_after_generate_calls)
    monkeypatch.setattr(
        experiment,
        "create_generation_components",
        lambda _model_name: (tokenizer, model),
    )
    monkeypatch.setattr(
        custom_sae,
        "create_generation_components",
        lambda _model_name: (tokenizer, model),
    )


def test_parse_format_completion_handles_word_hex_and_rgb() -> None:
    assert experiment.parse_format_completion("word", "Probably blue.").color_family == "blue"
    assert experiment.parse_format_completion("word", "Probably scarlet.").normalized_output == "scarlet"
    assert experiment.parse_format_completion("hex", "Try #abc").normalized_output == "#aabbcc"
    assert experiment.parse_format_completion("rgb", "255,0,0").color_family == "red"


def test_fit_transfer_accuracy_preserves_rare_consensus_labels() -> None:
    np = __import__("numpy")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold

    words = ["fire", "rose", "ocean", "sky", "forest", "moss", "orchid"]
    source_rows = [{"word": word} for word in words]
    target_rows = [{"word": word} for word in words]
    source_matrix = np.array(
        [
            [2.0, 0.0],
            [1.9, 0.1],
            [-2.0, 0.0],
            [-1.8, -0.1],
            [0.0, 2.0],
            [0.1, 1.8],
            [0.0, -2.0],
        ],
        dtype=np.float32,
    )
    target_matrix = source_matrix.copy()
    consensus_by_word = {
        "fire": "red",
        "rose": "red",
        "ocean": "blue",
        "sky": "blue",
        "forest": "green",
        "moss": "green",
        "orchid": "magenta",
    }

    accuracy = experiment._fit_transfer_accuracy(
        np=np,
        LogisticRegression=LogisticRegression,
        KFold=KFold,
        source_matrix=source_matrix,
        target_matrix=target_matrix,
        source_rows=source_rows,
        target_rows=target_rows,
        consensus_by_word=consensus_by_word,
    )

    assert accuracy is not None
    assert accuracy >= 0.5


def test_read_words_raises_if_large_limit_has_no_dictionary(monkeypatch) -> None:
    monkeypatch.setattr(experiment, "find_system_word_list", lambda: None)

    with pytest.raises(RuntimeError, match="Requested .* built-in words are available"):
        experiment._read_words(None, len(word_lists.default_words()) + 50)


def test_read_words_uses_system_dictionary_for_large_limit(tmp_path: Path, monkeypatch) -> None:
    system_words_path = tmp_path / "system_words.txt"
    system_words_path.write_text("violet\nsunrise\nviolet\nriver\nrose\npeach\n", encoding="utf-8")
    monkeypatch.setattr(experiment, "find_system_word_list", lambda: system_words_path)

    words, source = experiment._read_words(None, len(word_lists.default_words()) + 5)

    assert words == ["violet", "sunrise", "river", "rose", "peach"]
    assert source == str(system_words_path)


def test_read_words_uses_color_word_preset() -> None:
    words, source = experiment._read_words(None, 6, word_preset="color_words")

    assert words == ["red", "scarlet", "crimson", "carmine", "maroon", "burgundy"]
    assert source == "color_words"


def test_run_color_format_latent_experiment_writes_outputs_and_heartbeats(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_components(monkeypatch)
    word_list_path = tmp_path / "words.txt"
    word_list_path.write_text("\n".join(WORD_FAMILIES) + "\n", encoding="utf-8")

    summary = experiment.run_color_format_latent_experiment(
        output_dir=tmp_path,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        word_list_path=word_list_path,
        batch_size=2,
        device="cpu",
    )

    assert summary["consensus_word_count"] == 6
    assert summary["layers"] == [0, 1, 2]
    assert summary["best_cross_layer"] in {0, 1, 2}
    assert summary["best_cross_mean_accuracy"] is not None
    assert summary["best_within_schema_layer_by_format"]["word"] in {0, 1, 2}
    assert (tmp_path / "predictions_word.jsonl").exists()
    assert (tmp_path / "predictions_hex.jsonl").exists()
    assert (tmp_path / "predictions_rgb.jsonl").exists()
    assert (tmp_path / "consensus_labels.jsonl").exists()
    assert (tmp_path / "within_schema_probe_accuracy.jsonl").exists()
    assert (tmp_path / "cross_format_probe_transfer.jsonl").exists()
    assert (tmp_path / "layer_summary.jsonl").exists()
    assert (tmp_path / "shared_pca_grid.svg").exists()
    assert (tmp_path / "format_transfer_curve.svg").exists()
    assert (tmp_path / "final_results.json").exists()
    assert (tmp_path / "heartbeat_status.json").exists()
    assert (tmp_path / "heartbeat_events.jsonl").exists()
    assert (tmp_path / "manifest.json").exists()
    hidden = np.load(tmp_path / "hidden_states" / "word" / "layer_00.npy")
    assert hidden.shape == (6, 4)
    status = json.loads((tmp_path / "heartbeat_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "completed"
    events = [
        json.loads(line)
        for line in (tmp_path / "heartbeat_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(event["phase"] == "collect" for event in events)
    assert any(event["phase"] == "analyze" for event in events)
    transfer_rows = [
        json.loads(line)
        for line in (tmp_path / "cross_format_probe_transfer.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cross_rows = [
        row for row in transfer_rows if row["source_format"] == "word" and row["target_format"] == "hex"
    ]
    assert cross_rows
    assert all(row["accuracy"] is None or row["accuracy"] >= 0.5 for row in cross_rows)
    within_rows = [
        json.loads(line)
        for line in (tmp_path / "within_schema_probe_accuracy.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row["format"] == "word" and row["accuracy"] is not None for row in within_rows)


def test_run_color_format_patch_moves_target_toward_source_family(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_components(monkeypatch)
    word_list_path = tmp_path / "words.txt"
    word_list_path.write_text("\n".join(WORD_FAMILIES) + "\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    experiment.run_color_format_latent_experiment(
        output_dir=run_dir,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        word_list_path=word_list_path,
        batch_size=2,
        device="cpu",
    )

    pairs_path = tmp_path / "pairs.csv"
    pairs_path.write_text("fire,ocean\nrose,sky\n", encoding="utf-8")
    patch_dir = tmp_path / "patch"
    summary = experiment.run_color_format_patch(
        run_dir=run_dir,
        output_dir=patch_dir,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        source_format="word",
        target_format="hex",
        layer=1,
        pairs_path=pairs_path,
        batch_size=1,
        device="cpu",
    )

    assert summary["pair_count"] == 2
    assert summary["changed_rate"] == 1.0
    assert summary["moved_toward_source_rate"] == 1.0
    assert summary["patched_match_rate"] == 1.0
    rows = [
        json.loads(line)
        for line in (patch_dir / "patched_predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["baseline_target_family"] == "blue"
    assert rows[0]["patched_target_family"] == "red"
    assert rows[0]["source_family"] == "red"
    status = json.loads((patch_dir / "heartbeat_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "completed"
    assert (patch_dir / "final_results.json").exists()


def test_run_color_format_latent_experiment_resumes_from_batch_checkpoints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    word_list_path = tmp_path / "words.txt"
    word_list_path.write_text("\n".join(WORD_FAMILIES) + "\n", encoding="utf-8")
    _install_fake_components(monkeypatch, fail_after_generate_calls=1)

    try:
        experiment.run_color_format_latent_experiment(
            output_dir=tmp_path,
            model_name="Qwen/Qwen2.5-7B-Instruct",
            word_list_path=word_list_path,
            batch_size=2,
            device="cpu",
        )
    except RuntimeError as error:
        assert "simulated interruption" in str(error)
    else:
        raise AssertionError("expected the first run to interrupt")

    assert (tmp_path / "checkpoints" / "run_batches" / "word" / "batch_0001.predictions.jsonl").exists()
    _install_fake_components(monkeypatch)
    summary = experiment.run_color_format_latent_experiment(
        output_dir=tmp_path,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        word_list_path=word_list_path,
        batch_size=2,
        resume=True,
        device="cpu",
    )

    assert summary["consensus_word_count"] == 6
    events = [
        json.loads(line)
        for line in (tmp_path / "heartbeat_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any("from checkpoint" in event["message"] for event in events)


def test_export_final_results_copies_run_and_patch_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_components(monkeypatch)
    word_list_path = tmp_path / "words.txt"
    word_list_path.write_text("\n".join(WORD_FAMILIES) + "\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    patch_dir = tmp_path / "patch"
    export_dir = tmp_path / "export"

    experiment.run_color_format_latent_experiment(
        output_dir=run_dir,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        word_list_path=word_list_path,
        batch_size=2,
        device="cpu",
    )
    pairs_path = tmp_path / "pairs.csv"
    pairs_path.write_text("fire,ocean\nrose,sky\n", encoding="utf-8")
    experiment.run_color_format_patch(
        run_dir=run_dir,
        output_dir=patch_dir,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        source_format="word",
        target_format="hex",
        layer=1,
        pairs_path=pairs_path,
        batch_size=1,
        device="cpu",
    )

    payload = experiment.export_final_results(
        run_dir=run_dir,
        output_dir=export_dir,
        patch_dir=patch_dir,
    )

    assert payload["run_summary"]["model_name"] == "Qwen/Qwen2.5-7B-Instruct"
    assert payload["patch_summary"]["patched_match_rate"] == 1.0
    assert (export_dir / "final_results_bundle.json").exists()
    assert (export_dir / "run_final_results.json").exists()
    assert (export_dir / "patch_final_results.json").exists()


def test_run_color_logit_lens_experiment_writes_outputs_and_heartbeats(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_components(monkeypatch)
    word_list_path = tmp_path / "words.txt"
    word_list_path.write_text("\n".join(WORD_FAMILIES) + "\n", encoding="utf-8")

    summary = logit_lens.run_color_logit_lens_experiment(
        output_dir=tmp_path,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        word_list_path=word_list_path,
        batch_size=2,
        device="cpu",
    )

    assert summary["layers"] == [0, 1, 2]
    assert summary["best_color_family_layer_by_format"]["word"] in {0, 1, 2}
    assert (tmp_path / "logit_lens_rows.jsonl").exists()
    assert (tmp_path / "layer_summary.jsonl").exists()
    assert (tmp_path / "top_token_snapshots.jsonl").exists()
    assert (tmp_path / "logit_lens_curve.svg").exists()
    assert (tmp_path / "interpretation.json").exists()
    assert (tmp_path / "interpretation.md").exists()
    assert (tmp_path / "final_results.json").exists()
    status = json.loads((tmp_path / "heartbeat_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "completed"
    layer_rows = [
        json.loads(line)
        for line in (tmp_path / "layer_summary.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hex_rows = [row for row in layer_rows if row["format"] == "hex"]
    assert hex_rows
    assert any(row["mean_hex_mass"] is not None for row in hex_rows)
    interpretation = json.loads((tmp_path / "interpretation.json").read_text(encoding="utf-8"))
    assert interpretation["semantic_color_onset_by_format"]["hex"] in {0, 1, 2}
    assert interpretation["rendering_onset_layers"]["hex"] in {0, 1, 2}
    assert interpretation["headline_findings"]


def test_run_color_word_basis_experiment_writes_stage_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_components(monkeypatch)
    word_list_path = tmp_path / "anchor_words.txt"
    word_list_path.write_text("\n".join(WORD_FAMILIES) + "\n", encoding="utf-8")

    summary = experiment.run_color_word_basis_experiment(
        output_dir=tmp_path / "basis",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        word_list_path=word_list_path,
        limit=6,
        sae_layer=1,
        batch_size=2,
        sae_activation_batch_size=2,
        sae_train_batch_size=2,
        sae_dictionary_multiplier=2,
        sae_top_k=2,
        sae_epochs=2,
        sae_max_length=96,
        device="cpu",
    )

    basis_dir = tmp_path / "basis"
    assert summary["run_summary"]["best_cross_layer"] in {0, 1, 2}
    assert summary["sae_training_summary"]["layer"] == 1
    assert summary["sae_analysis_summary"]["format_name"] == "word"
    assert (basis_dir / "run" / "shared_pca_grid.svg").exists()
    assert (basis_dir / "logit_lens" / "interpretation.json").exists()
    assert (basis_dir / "sae_train_word" / "sae_checkpoint.pt").exists()
    assert (basis_dir / "sae_analysis_word" / "schema_label_rankings.json").exists()
    assert (basis_dir / "summary.json").exists()
    assert (basis_dir / "final_results.json").exists()
    status = json.loads((basis_dir / "heartbeat_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "completed"


def test_run_color_logit_lens_experiment_resumes_from_batch_checkpoints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    word_list_path = tmp_path / "words.txt"
    word_list_path.write_text("\n".join(WORD_FAMILIES) + "\n", encoding="utf-8")
    _install_fake_components(monkeypatch, fail_after_generate_calls=1)

    try:
        logit_lens.run_color_logit_lens_experiment(
            output_dir=tmp_path,
            model_name="Qwen/Qwen2.5-7B-Instruct",
            word_list_path=word_list_path,
            batch_size=2,
            device="cpu",
        )
    except RuntimeError as error:
        assert "simulated interruption" in str(error)
    else:
        raise AssertionError("expected the first logit lens run to interrupt")

    assert (
        tmp_path
        / "checkpoints"
        / "logit_lens_batches"
        / "word"
        / "batch_0001.rows.jsonl"
    ).exists()
    _install_fake_components(monkeypatch)
    summary = logit_lens.run_color_logit_lens_experiment(
        output_dir=tmp_path,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        word_list_path=word_list_path,
        batch_size=2,
        resume=True,
        device="cpu",
    )

    assert summary["best_color_family_layer_by_format"]["word"] in {0, 1, 2}
    events = [
        json.loads(line)
        for line in (tmp_path / "heartbeat_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any("from checkpoint" in event["message"] for event in events)


def test_export_final_results_copies_logit_lens_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_components(monkeypatch)
    word_list_path = tmp_path / "words.txt"
    word_list_path.write_text("\n".join(WORD_FAMILIES) + "\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    logit_dir = tmp_path / "logit"
    export_dir = tmp_path / "export"

    experiment.run_color_format_latent_experiment(
        output_dir=run_dir,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        word_list_path=word_list_path,
        batch_size=2,
        device="cpu",
    )
    logit_lens.run_color_logit_lens_experiment(
        output_dir=logit_dir,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        word_list_path=word_list_path,
        batch_size=2,
        device="cpu",
    )

    payload = experiment.export_final_results(
        run_dir=run_dir,
        output_dir=export_dir,
        logit_lens_dir=logit_dir,
    )

    assert payload["logit_lens_summary"]["model_name"] == "Qwen/Qwen2.5-7B-Instruct"
    assert (export_dir / "logit_lens_final_results.json").exists()
    assert (export_dir / "logit_lens_interpretation.json").exists()
    assert (export_dir / "logit_lens_interpretation.md").exists()
    assert (export_dir / "logit_lens_logit_lens_curve.svg").exists()


def test_summarize_logit_lens_run_rewrites_interpretation_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_components(monkeypatch)
    word_list_path = tmp_path / "words.txt"
    word_list_path.write_text("\n".join(WORD_FAMILIES) + "\n", encoding="utf-8")

    logit_lens.run_color_logit_lens_experiment(
        output_dir=tmp_path,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        word_list_path=word_list_path,
        batch_size=2,
        device="cpu",
    )
    (tmp_path / "interpretation.json").unlink()
    (tmp_path / "interpretation.md").unlink()

    interpretation = logit_lens.summarize_logit_lens_run(tmp_path)

    assert interpretation["best_color_family_layer_by_format"]["word"] in {0, 1, 2}
    assert (tmp_path / "interpretation.json").exists()
    assert (tmp_path / "interpretation.md").exists()
    final_results = json.loads((tmp_path / "final_results.json").read_text(encoding="utf-8"))
    assert final_results["interpretation"]["headline_findings"]
