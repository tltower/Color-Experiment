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

from color_latent_lab import custom_sae


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    unk_token = "<unk>"
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "right"

    def __init__(self) -> None:
        self._word_to_id: dict[str, int] = {}

    def _word_id(self, text: str) -> int:
        word = text.split("word ", 1)[1].split("?", 1)[0].strip().lower()
        if word not in self._word_to_id:
            self._word_to_id[word] = len(self._word_to_id) + 10
        return self._word_to_id[word]

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
        rows = [[2, self._word_id(text), 3] for text in texts]
        return {
            "input_ids": torch.tensor(rows, dtype=torch.int64),
            "attention_mask": torch.ones((len(rows), 3), dtype=torch.int64),
        }


class FakeModel(torch.nn.Module):
    config = type("Config", (), {"num_hidden_layers": 3, "hidden_size": 6})()

    def to(self, _device: object) -> "FakeModel":
        return self

    def eval(self) -> "FakeModel":
        return self

    def forward(self, **kwargs: torch.Tensor):
        input_ids = kwargs["input_ids"].float()
        word_ids = input_ids[:, 1]
        hidden_states = []
        for layer in range(4):
            base = input_ids.unsqueeze(-1).repeat(1, 1, 6)
            base[:, -1, 0] = word_ids + layer
            base[:, -1, 1] = (word_ids % 2) * (layer + 1.0)
            base[:, -1, 2] = (word_ids % 3) * 0.5
            base[:, -1, 3] = word_ids / 10.0
            base[:, -1, 4] = layer
            base[:, -1, 5] = 1.0
            hidden_states.append(base)
        return type("Output", (), {"hidden_states": tuple(hidden_states)})()


def test_run_color_sae_training_writes_checkpoint_and_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        custom_sae,
        "create_generation_components",
        lambda _model_name: (FakeTokenizer(), FakeModel()),
    )

    word_list_path = tmp_path / "words.txt"
    word_list_path.write_text(
        "\n".join(
            ["apple", "river", "forest", "brick", "ocean", "sun", "night", "rose", "glass", "storm"]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "train"
    summary = custom_sae.run_color_sae_training(
        output_dir=output_dir,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        layer=1,
        prompt_format="hex",
        word_list_path=word_list_path,
        limit=10,
        max_length=64,
        activation_batch_size=4,
        train_batch_size=4,
        device="cpu",
        dictionary_multiplier=2,
        top_k=2,
        epochs=2,
        learning_rate=1e-2,
        validation_fraction=0.2,
        reuse_cached_activations=False,
    )

    assert summary["layer"] == 1
    assert summary["activation_count"] == 10
    assert summary["input_dim"] == 6
    assert (output_dir / "sae_checkpoint.pt").exists()
    assert (output_dir / "feature_usage_summary.json").exists()
    assert (output_dir / "training_curve.jsonl").exists()
    assert (output_dir / "final_results.json").exists()
    assert (output_dir / "checkpoints" / "latest_sae_checkpoint.pt").exists()
    assert (output_dir / "checkpoints" / "training_state.json").exists()


def test_run_color_sae_feature_analysis_writes_family_rankings_and_directions(
    tmp_path: Path,
) -> None:
    model = custom_sae.SparseAutoencoder(input_dim=6, dictionary_size=4, top_k=None)
    with torch.no_grad():
        model.input_bias.zero_()
        model.encoder.weight.zero_()
        model.encoder.bias.zero_()
        model.decoder.weight.zero_()
        model.encoder.weight[0, 0] = 2.0
        model.encoder.weight[1, 1] = 2.0
        model.encoder.weight[2, 2] = 1.0
        model.encoder.weight[3, 3] = 1.0
        model.decoder.weight[0, 0] = 1.0
        model.decoder.weight[1, 1] = 1.0
        model.decoder.weight[2, 2] = 1.0
        model.decoder.weight[3, 3] = 1.0
        model.normalize_decoder()

    checkpoint_path = tmp_path / "sae_checkpoint.pt"
    torch.save(
        {
            "config": {
                "dictionary_size": 4,
                "input_dim": 6,
                "layer": 1,
                "model_name": "Qwen/Qwen2.5-7B-Instruct",
                "prompt_format": "hex",
                "prompt_template": custom_sae.FORMAT_PROMPTS["hex"],
                "top_k": None,
            },
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    color_run_dir = tmp_path / "color_run"
    for format_name in ("word", "hex", "rgb"):
        hidden_dir = color_run_dir / "hidden_states" / format_name
        hidden_dir.mkdir(parents=True, exist_ok=True)
        activations = np.array(
            [
                [2.0, 0.2, 0.0, 0.0, 0.0, 0.0],
                [1.8, 0.1, 0.0, 0.0, 0.0, 0.0],
                [2.1, 0.3, 0.0, 0.0, 0.0, 0.0],
                [1.9, 0.2, 0.0, 0.0, 0.0, 0.0],
                [0.1, 2.1, 0.0, 0.0, 0.0, 0.0],
                [0.2, 1.8, 0.0, 0.0, 0.0, 0.0],
                [0.3, 2.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 1.9, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        np.save(hidden_dir / "layer_01.npy", activations.astype(np.float16))
        prediction_rows = [
            {"word": "fire", "normalized_output": "red", "color_family": "red", "temperature": "warm", "format": format_name},
            {"word": "rose", "normalized_output": "red", "color_family": "red", "temperature": "warm", "format": format_name},
            {"word": "sun", "normalized_output": "red", "color_family": "red", "temperature": "warm", "format": format_name},
            {"word": "ember", "normalized_output": "red", "color_family": "red", "temperature": "warm", "format": format_name},
            {"word": "ocean", "normalized_output": "blue", "color_family": "blue", "temperature": "cool", "format": format_name},
            {"word": "night", "normalized_output": "blue", "color_family": "blue", "temperature": "cool", "format": format_name},
            {"word": "river", "normalized_output": "blue", "color_family": "blue", "temperature": "cool", "format": format_name},
            {"word": "sky", "normalized_output": "blue", "color_family": "blue", "temperature": "cool", "format": format_name},
        ]
        (color_run_dir / f"predictions_{format_name}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in prediction_rows) + "\n",
            encoding="utf-8",
        )

    output_dir = tmp_path / "analysis"
    summary = custom_sae.run_color_sae_feature_analysis(
        sae_checkpoint_path=checkpoint_path,
        color_run_dir=color_run_dir,
        output_dir=output_dir,
        layer=1,
        format_name="all",
        batch_size=4,
        device="cpu",
    )

    assert summary["layer"] == 1
    assert summary["warm_cool"] is not None
    assert summary["red_blue"] is not None
    assert "red" in summary["family_rankings"]
    assert "blue" in summary["family_rankings"]
    assert (output_dir / "encoded_features.npy").exists()
    assert (output_dir / "family_rankings.json").exists()
    assert (output_dir / "warm_cool_decoder_direction.npy").exists()
    assert (output_dir / "red_blue_decoder_direction.npy").exists()
    assert (output_dir / "family_decoder_directions" / "red_decoder_direction.npy").exists()
    assert (output_dir / "feature_examples.jsonl").exists()
    assert (output_dir / "final_results.json").exists()
