# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import color_latent_lab.custom_sae as custom_sae
import color_latent_lab.sae_geometry as sae_geometry
import color_latent_lab.word_set_sae as word_set_sae


class FakeWordSetTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    unk_token = "<unk>"
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "left"

    def __init__(self) -> None:
        self._next_prompt_id = 10
        self.id_to_word: dict[int, str] = {}

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
        rows: list[list[int]] = []
        for text in texts:
            prompt_id = self._next_prompt_id
            self._next_prompt_id += 1
            self.id_to_word[prompt_id] = text
            rows.append([2, prompt_id])
        return {
            "input_ids": torch.tensor(rows, dtype=torch.int64),
            "attention_mask": torch.ones((len(rows), 2), dtype=torch.int64),
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
        return "".join(self.id_to_word.get(int(token_id), f"<tok:{int(token_id)}>") for token_id in token_ids)


class FakeWordSetModel(torch.nn.Module):
    config = type("Config", (), {"num_hidden_layers": 2, "hidden_size": 4})()

    def __init__(self, tokenizer: FakeWordSetTokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def _word_vector(self, word: str, prompt_id: int, device: torch.device) -> torch.Tensor:
        base = {
            "red": torch.tensor([2.0, 0.2, 0.0, 0.0], device=device),
            "blue": torch.tensor([-2.0, 0.2, 0.0, 0.0], device=device),
            "green": torch.tensor([0.0, 2.0, 0.0, 0.0], device=device),
        }[word]
        return base + torch.tensor([0.0, 0.0, float(prompt_id) / 100.0, 1.0], device=device)

    def forward(self, **kwargs: torch.Tensor):
        input_ids = kwargs["input_ids"]
        device = input_ids.device
        batch_size, sequence_length = input_ids.shape
        hidden0 = torch.zeros((batch_size, sequence_length, 4), dtype=torch.float32, device=device)
        for row_index, prompt_id in enumerate(input_ids[:, -1].tolist()):
            word = self.tokenizer.id_to_word[int(prompt_id)]
            hidden0[row_index, -1, :] = self._word_vector(word, int(prompt_id), device)
        hidden1 = hidden0.clone()
        hidden1[:, :, 0] = hidden1[:, :, 0] * 1.10
        hidden1[:, :, 2] = hidden1[:, :, 2] + 0.15
        hidden2 = hidden1.clone()
        hidden2[:, :, 1] = hidden2[:, :, 1] * 1.20
        hidden2[:, :, 3] = hidden2[:, :, 3] + 0.10
        return type("Output", (), {"hidden_states": (hidden0, hidden1, hidden2)})()


def _install_fake_components(monkeypatch) -> None:
    tokenizer = FakeWordSetTokenizer()
    model = FakeWordSetModel(tokenizer)
    monkeypatch.setattr(sae_geometry, "create_generation_components", lambda _model_name: (tokenizer, model))


def _write_fake_sae_repo(root: Path, *, layers: tuple[int, ...]) -> None:
    for layer in layers:
        layer_dir = root / f"resid_post_layer_{layer}" / "trainer_0"
        layer_dir.mkdir(parents=True, exist_ok=True)
        model = custom_sae.SparseAutoencoder(input_dim=4, dictionary_size=5, top_k=None)
        with torch.no_grad():
            model.input_bias.zero_()
            model.encoder.weight.zero_()
            model.encoder.bias.zero_()
            model.decoder.weight.zero_()
            model.encoder.weight[0, 0] = 1.0
            model.encoder.weight[1, 0] = -1.0
            model.encoder.weight[2, 1] = 1.0
            model.encoder.weight[3, 2] = 1.0
            model.encoder.weight[4, 3] = 1.0
            model.decoder.weight[0, 0] = 1.0
            model.decoder.weight[0, 1] = -1.0
            model.decoder.weight[1, 2] = 1.0
            model.decoder.weight[2, 3] = 1.0
            model.decoder.weight[3, 4] = 1.0
        torch.save(model.state_dict(), layer_dir / "ae.pt")
        (layer_dir / "config.json").write_text(
            json.dumps({"dictionary_size": 5, "input_dim": 4}),
            encoding="utf-8",
        )


def test_leave_one_out_feature_vectors_matches_expected() -> None:
    values = np.array(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )
    mean_others, leave_one_out = word_set_sae._leave_one_out_feature_vectors(np, values)
    expected_mean_others = np.array(
        [
            [1.5, 3.0],
            [2.0, 2.0],
            [0.5, 1.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(mean_others, expected_mean_others)
    assert np.allclose(leave_one_out, values - expected_mean_others)


def test_run_word_set_sae_feature_experiment_writes_cosine_matrices(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _install_fake_components(monkeypatch)
    monkeypatch.setattr(word_set_sae, "COMMON_COLOR_FAMILY_WORDS", ("red", "blue", "green"))
    sae_repo = tmp_path / "fake_sae_repo"
    _write_fake_sae_repo(sae_repo, layers=(0, 1))

    output_dir = tmp_path / "out"
    summary = word_set_sae.run_word_set_sae_feature_experiment(
        output_dir=output_dir,
        model_name="fake-model",
        sae_repo_id_or_path=str(sae_repo),
        sae_layers=(0, 1),
        batch_size=4,
        encode_batch_size=8,
        max_length=8,
        device="cpu",
    )

    assert summary["words"] == ["red", "blue", "green"]
    feature_vectors = np.load(output_dir / "layer_00" / "feature_vectors.npy")
    mean_other = np.load(output_dir / "layer_00" / "mean_other_feature_vectors.npy")
    leave_one_out = np.load(output_dir / "layer_00" / "leave_one_out_feature_vectors.npy")
    similarity = np.load(output_dir / "layer_00" / "cosine_similarity_matrix.npy")

    assert feature_vectors.shape == (3, 5)
    assert np.allclose(leave_one_out, feature_vectors - mean_other)

    norms = np.linalg.norm(leave_one_out, axis=1, keepdims=True) + 1e-8
    expected_similarity = (leave_one_out / norms) @ (leave_one_out / norms).T
    assert np.allclose(similarity, expected_similarity)

    matrix_payload = json.loads((output_dir / "layer_00" / "cosine_similarity_matrix.json").read_text(encoding="utf-8"))
    assert matrix_payload["words"] == ["red", "blue", "green"]
    assert len(matrix_payload["matrix"]) == 3
    assert (output_dir / "layer_01" / "cosine_similarity_matrix.json").exists()
    captured = capsys.readouterr()
    assert "Layer 0 cosine similarity matrix" in captured.out
    assert "red" in captured.out
    assert "blue" in captured.out
    assert "green" in captured.out
