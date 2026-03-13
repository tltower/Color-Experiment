from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .color_formats import FORMAT_PROMPTS
from .model_utils import _non_padding_last_positions, _render_prompt, _resolve_device
from .run_support import (
    HeartbeatRecorder,
    _read_prediction_rows,
    _utc_now,
    _write_json,
    _write_jsonl,
)
from .hf import create_generation_components
from .word_lists import default_words, find_system_word_list, preset_words, read_word_file


def load_training_words(
    *,
    word_list_path: Path | None,
    word_preset: str,
    limit: int,
) -> list[str]:
    if word_list_path is None:
        if word_preset == "color_words":
            return preset_words("color_words", limit=limit)
        source_path = find_system_word_list()
        if source_path is None:
            default_seed = default_words()
            if limit <= len(default_seed):
                return preset_words(word_preset, limit=limit)
            raise RuntimeError(
                f"Requested {limit} words, but only {len(default_seed)} built-in words are available. "
                "Pass `--word-list-path` or install a system dictionary."
            )
    else:
        source_path = word_list_path

    words = read_word_file(source_path, limit=limit)
    if not words:
        raise ValueError(f"No valid words loaded from {source_path}")
    return words


class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        dictionary_size: int,
        top_k: int | None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.dictionary_size = dictionary_size
        self.top_k = top_k
        self.input_bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder = nn.Linear(input_dim, dictionary_size)
        self.decoder = nn.Linear(dictionary_size, input_dim, bias=False)
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        self.normalize_decoder()

    def normalize_decoder(self) -> None:
        with torch.no_grad():
            norms = self.decoder.weight.data.norm(dim=0, keepdim=True).clamp_min(1e-8)
            self.decoder.weight.data.div_(norms)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        centered = inputs - self.input_bias
        features = torch.relu(self.encoder(centered))
        if self.top_k is not None and 0 < self.top_k < self.dictionary_size:
            values, indices = torch.topk(features, k=self.top_k, dim=-1)
            sparse = torch.zeros_like(features)
            sparse.scatter_(1, indices, values)
            features = sparse
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features) + self.input_bias

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(inputs)
        reconstruction = self.decode(features)
        return reconstruction, features

    def decoder_vectors(self) -> torch.Tensor:
        return self.decoder.weight.detach().cpu().T


def _checkpoint_paths(output_dir: Path) -> dict[str, Path]:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return {
        "best_model": output_dir / "sae_checkpoint.pt",
        "latest_model": checkpoint_dir / "latest_sae_checkpoint.pt",
        "state": checkpoint_dir / "training_state.json",
    }


def _save_checkpoint(
    *,
    path: Path,
    model: SparseAutoencoder,
    config: dict[str, Any],
    optimizer_state: dict[str, Any] | None = None,
    epoch: int | None = None,
    best_validation_loss: float | None = None,
) -> None:
    payload = {
        "best_validation_loss": best_validation_loss,
        "config": config,
        "epoch": epoch,
        "optimizer_state_dict": optimizer_state,
        "saved_at_utc": _utc_now(),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def load_sparse_autoencoder_checkpoint(
    path: Path,
    *,
    device: str = "cpu",
) -> tuple[SparseAutoencoder, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    config = dict(payload["config"])
    model = SparseAutoencoder(
        input_dim=int(config["input_dim"]),
        dictionary_size=int(config["dictionary_size"]),
        top_k=config.get("top_k"),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, config


def _capture_layer_activations(
    *,
    model_name: str,
    words: list[str],
    prompt_template: str,
    layer: int,
    max_length: int,
    batch_size: int,
    device: str,
    heartbeat: HeartbeatRecorder,
) -> np.ndarray:
    tokenizer, model = create_generation_components(model_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    runtime_device = _resolve_device(torch, device)
    model.to(runtime_device)
    model.eval()

    rendered_prompts = [_render_prompt(tokenizer, prompt_template.format(word=word)) for word in words]
    cached_vectors: list[np.ndarray] = []
    for batch_start in range(0, len(words), batch_size):
        batch_prompts = rendered_prompts[batch_start : batch_start + batch_size]
        tokenized = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokenized = {key: value.to(runtime_device) for key, value in tokenized.items()}
        heartbeat.update(
            phase="capture",
            message="Caching SAE training activations",
            layer=layer,
            processed_words=batch_start,
            total_words=len(words),
        )
        with torch.no_grad():
            outputs = model(**tokenized)
        hidden_states = outputs.hidden_states
        if layer < 0 or layer >= len(hidden_states):
            raise ValueError(f"Requested layer {layer} outside valid range 0..{len(hidden_states) - 1}")
        last_positions = _non_padding_last_positions(tokenized["attention_mask"])
        for batch_index in range(len(batch_prompts)):
            cached_vectors.append(
                hidden_states[layer][batch_index, last_positions[batch_index]]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    return np.vstack(cached_vectors)


def _split_train_validation(
    activations: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if activations.shape[0] < 4:
        return activations, activations
    rng = np.random.default_rng(seed)
    indices = np.arange(activations.shape[0])
    rng.shuffle(indices)
    validation_count = max(1, int(round(activations.shape[0] * validation_fraction)))
    validation_count = min(validation_count, activations.shape[0] - 1)
    validation_indices = indices[:validation_count]
    train_indices = indices[validation_count:]
    return activations[train_indices], activations[validation_indices]


def _evaluate_autoencoder(
    *,
    model: SparseAutoencoder,
    activations: np.ndarray,
    batch_size: int,
    device: torch.device,
    l1_coefficient: float,
) -> dict[str, float]:
    mse_total = 0.0
    l1_total = 0.0
    active_total = 0.0
    sample_count = 0
    with torch.no_grad():
        for start in range(0, activations.shape[0], batch_size):
            batch = torch.from_numpy(activations[start : start + batch_size]).to(device).float()
            reconstruction, features = model(batch)
            mse = F.mse_loss(reconstruction, batch, reduction="sum").item()
            l1_value = features.abs().sum().item()
            active = (features > 0).float().sum().item()
            mse_total += mse
            l1_total += l1_value
            active_total += active
            sample_count += int(batch.shape[0])
    denominator = max(sample_count, 1)
    return {
        "avg_active_features": float(active_total / denominator),
        "l1": float(l1_total / denominator),
        "loss": float((mse_total / denominator) + l1_coefficient * (l1_total / denominator)),
        "mse": float(mse_total / denominator),
    }


def _feature_usage_summary(
    *,
    model: SparseAutoencoder,
    activations: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    feature_sum = None
    feature_count = None
    sample_count = 0
    with torch.no_grad():
        for start in range(0, activations.shape[0], batch_size):
            batch = torch.from_numpy(activations[start : start + batch_size]).to(device).float()
            features = model.encode(batch)
            batch_sum = features.sum(dim=0).detach().cpu().numpy()
            batch_count = (features > 0).sum(dim=0).detach().cpu().numpy()
            if feature_sum is None:
                feature_sum = batch_sum
                feature_count = batch_count
            else:
                feature_sum += batch_sum
                feature_count += batch_count
            sample_count += int(batch.shape[0])
    if feature_sum is None or feature_count is None:
        raise RuntimeError("No feature usage statistics were produced.")
    firing_rates = feature_count.astype(np.float64) / max(sample_count, 1)
    mean_activations = feature_sum.astype(np.float64) / max(sample_count, 1)
    ranking = np.argsort(firing_rates)[::-1][:20]
    return {
        "active_feature_count": int((feature_count > 0).sum()),
        "mean_active_features_per_example": float(feature_count.sum() / max(sample_count, 1)),
        "sample_count": sample_count,
        "top_features_by_firing_rate": [
            {
                "feature": int(index),
                "firing_rate": float(firing_rates[index]),
                "mean_activation": float(mean_activations[index]),
            }
            for index in ranking
        ],
    }


def _write_training_final_results(output_dir: Path, *, summary: dict[str, Any]) -> None:
    payload = {
        "kind": "custom_sae_training",
        "key_artifacts": {
            "checkpoint": "sae_checkpoint.pt",
            "feature_usage": "feature_usage_summary.json",
            "heartbeat_status": "heartbeat_status.json",
            "summary": "summary.json",
            "training_curve": "training_curve.jsonl",
        },
        "summary": summary,
    }
    _write_json(output_dir / "final_results.json", payload)


def _write_analysis_final_results(output_dir: Path, *, summary: dict[str, Any]) -> None:
    payload = {
        "kind": "custom_sae_analysis",
        "key_artifacts": {
            "encoded_features": "encoded_features.npy",
            "family_rankings": "family_rankings.json",
            "feature_examples": "feature_examples.jsonl",
            "heartbeat_status": "heartbeat_status.json",
            "schema_label_rankings": "schema_label_rankings.json",
            "summary": "summary.json",
        },
        "summary": summary,
    }
    _write_json(output_dir / "final_results.json", payload)


def run_color_sae_training(
    *,
    output_dir: Path,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    layer: int,
    prompt_format: str = "hex",
    prompt_template: str | None = None,
    word_list_path: Path | None = None,
    word_preset: str = "default",
    limit: int = 10000,
    max_length: int = 256,
    activation_batch_size: int = 32,
    train_batch_size: int = 256,
    device: str = "auto",
    dictionary_multiplier: int = 8,
    dictionary_size: int | None = None,
    top_k: int | None = 32,
    epochs: int = 8,
    learning_rate: float = 1e-3,
    l1_coefficient: float = 1e-4,
    validation_fraction: float = 0.1,
    seed: int = 17,
    reuse_cached_activations: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    if prompt_template is None:
        if prompt_format not in FORMAT_PROMPTS:
            raise ValueError(f"Unsupported prompt format {prompt_format!r}")
        prompt_template = FORMAT_PROMPTS[prompt_format]

    output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = HeartbeatRecorder(output_dir, label="sae-train")
    heartbeat.write_manifest(
        activation_batch_size=activation_batch_size,
        command="sae-train",
        device=device,
        dictionary_multiplier=dictionary_multiplier,
        dictionary_size=dictionary_size,
        epochs=epochs,
        l1_coefficient=l1_coefficient,
        layer=layer,
        learning_rate=learning_rate,
        limit=limit,
        max_length=max_length,
        model_name=model_name,
        prompt_format=prompt_format,
        resume=resume,
        top_k=top_k,
        train_batch_size=train_batch_size,
        validation_fraction=validation_fraction,
        word_list_path=None if word_list_path is None else str(word_list_path),
        word_preset=word_preset,
    )
    phase = "setup"
    try:
        words = load_training_words(word_list_path=word_list_path, word_preset=word_preset, limit=limit)
        (output_dir / "words.txt").write_text("\n".join(words) + "\n", encoding="utf-8")
        runtime_device = _resolve_device(torch, device)
        config = {
            "dictionary_multiplier": dictionary_multiplier,
            "dictionary_size": dictionary_size,
            "epochs": epochs,
            "l1_coefficient": l1_coefficient,
            "layer": layer,
            "learning_rate": learning_rate,
            "limit": limit,
            "max_length": max_length,
            "model_name": model_name,
            "prompt_format": prompt_format,
            "prompt_template": prompt_template,
            "seed": seed,
            "top_k": top_k,
            "validation_fraction": validation_fraction,
            "word_count": len(words),
            "word_preset": word_preset,
        }
        paths = _checkpoint_paths(output_dir)
        if resume and paths["state"].exists():
            state = json.loads(paths["state"].read_text(encoding="utf-8"))
            if state.get("config") != config:
                raise ValueError("SAE resume config mismatch; use a fresh output dir.")
        else:
            state = {"best_validation_loss": None, "config": config, "completed_epochs": 0}
            _write_json(paths["state"], state)

        activations_path = output_dir / f"layer_{layer:02d}_{prompt_format}_activations.npy"
        if reuse_cached_activations and activations_path.exists():
            activations = np.load(activations_path).astype(np.float32)
            heartbeat.update(
                phase="capture",
                message="Loaded cached SAE activations",
                path=str(activations_path),
                sample_count=int(activations.shape[0]),
            )
        else:
            activations = _capture_layer_activations(
                model_name=model_name,
                words=words,
                prompt_template=prompt_template,
                layer=layer,
                max_length=max_length,
                batch_size=activation_batch_size,
                device=device,
                heartbeat=heartbeat,
            ).astype(np.float32)
            np.save(activations_path, activations.astype(np.float16))

        train_activations, validation_activations = _split_train_validation(
            activations,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        input_dim = int(activations.shape[1])
        resolved_dictionary_size = (
            int(dictionary_size)
            if dictionary_size is not None
            else int(max(input_dim * dictionary_multiplier, max(top_k or 0, 4)))
        )
        torch.manual_seed(seed)
        model = SparseAutoencoder(
            input_dim=input_dim,
            dictionary_size=resolved_dictionary_size,
            top_k=top_k,
        )
        model.to(runtime_device)
        with torch.no_grad():
            mean_activation = torch.from_numpy(train_activations.mean(axis=0)).to(runtime_device).float()
            model.input_bias.copy_(mean_activation)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        start_epoch = 1
        if resume and paths["latest_model"].exists():
            payload = torch.load(paths["latest_model"], map_location=runtime_device)
            model.load_state_dict(payload["state_dict"])
            if payload.get("optimizer_state_dict") is not None:
                optimizer.load_state_dict(payload["optimizer_state_dict"])
            start_epoch = int(payload.get("epoch", 0)) + 1
            state["best_validation_loss"] = payload.get("best_validation_loss")
            heartbeat.update(
                phase="training",
                message=f"Resuming SAE training from epoch {start_epoch}",
                completed_epochs=int(payload.get("epoch", 0)),
                total_epochs=epochs,
            )

        curve_path = output_dir / "training_curve.jsonl"
        curve_rows: list[dict[str, Any]] = []
        if resume and curve_path.exists():
            curve_rows = _read_prediction_rows(curve_path)

        best_validation_loss = state.get("best_validation_loss")
        for epoch in range(start_epoch, epochs + 1):
            permutation = np.random.default_rng(seed + epoch).permutation(train_activations.shape[0])
            model.train()
            for start in range(0, len(permutation), train_batch_size):
                batch_indices = permutation[start : start + train_batch_size]
                batch = torch.from_numpy(train_activations[batch_indices]).to(runtime_device).float()
                optimizer.zero_grad(set_to_none=True)
                reconstruction, features = model(batch)
                mse_loss = F.mse_loss(reconstruction, batch)
                l1_loss = features.abs().mean()
                loss = mse_loss + l1_coefficient * l1_loss
                loss.backward()
                optimizer.step()
                model.normalize_decoder()
            model.eval()
            train_metrics = _evaluate_autoencoder(
                model=model,
                activations=train_activations,
                batch_size=train_batch_size,
                device=runtime_device,
                l1_coefficient=l1_coefficient,
            )
            validation_metrics = _evaluate_autoencoder(
                model=model,
                activations=validation_activations,
                batch_size=train_batch_size,
                device=runtime_device,
                l1_coefficient=l1_coefficient,
            )
            row = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            curve_rows.append(row)
            _write_jsonl(curve_path, curve_rows)
            current_validation_loss = float(validation_metrics["loss"])
            if best_validation_loss is None or current_validation_loss < float(best_validation_loss):
                best_validation_loss = current_validation_loss
                _save_checkpoint(
                    path=paths["best_model"],
                    model=model,
                    config={
                        "dictionary_size": resolved_dictionary_size,
                        "input_dim": input_dim,
                        "layer": layer,
                        "model_name": model_name,
                        "prompt_format": prompt_format,
                        "prompt_template": prompt_template,
                        "top_k": top_k,
                    },
                    epoch=epoch,
                    best_validation_loss=best_validation_loss,
                )
            _save_checkpoint(
                path=paths["latest_model"],
                model=model,
                config={
                    "dictionary_size": resolved_dictionary_size,
                    "input_dim": input_dim,
                    "layer": layer,
                    "model_name": model_name,
                    "prompt_format": prompt_format,
                    "prompt_template": prompt_template,
                    "top_k": top_k,
                },
                optimizer_state=optimizer.state_dict(),
                epoch=epoch,
                best_validation_loss=best_validation_loss,
            )
            state["best_validation_loss"] = best_validation_loss
            state["completed_epochs"] = epoch
            _write_json(paths["state"], state)
            heartbeat.update(
                phase="training",
                message="Completed SAE training epoch",
                epoch=epoch,
                epochs=epochs,
                train_loss=train_metrics["loss"],
                validation_loss=validation_metrics["loss"],
            )

        model, _checkpoint_config = load_sparse_autoencoder_checkpoint(
            paths["best_model"],
            device=str(runtime_device),
        )
        usage_summary = _feature_usage_summary(
            model=model,
            activations=activations,
            batch_size=train_batch_size,
            device=runtime_device,
        )
        _write_json(output_dir / "feature_usage_summary.json", usage_summary)
        final_train_metrics = _evaluate_autoencoder(
            model=model,
            activations=train_activations,
            batch_size=train_batch_size,
            device=runtime_device,
            l1_coefficient=l1_coefficient,
        )
        final_validation_metrics = _evaluate_autoencoder(
            model=model,
            activations=validation_activations,
            batch_size=train_batch_size,
            device=runtime_device,
            l1_coefficient=l1_coefficient,
        )
        summary = {
            "activation_count": int(activations.shape[0]),
            "checkpoint_path": str(paths["best_model"]),
            "device": str(runtime_device),
            "dictionary_size": resolved_dictionary_size,
            "epochs": epochs,
            "feature_usage": usage_summary,
            "generated_at_utc": _utc_now(),
            "input_dim": input_dim,
            "l1_coefficient": l1_coefficient,
            "layer": layer,
            "learning_rate": learning_rate,
            "model_name": model_name,
            "prompt_format": prompt_format,
            "resume": resume,
            "top_k": top_k,
            "train_metrics": final_train_metrics,
            "validation_metrics": final_validation_metrics,
            "word_count": len(words),
            "word_preset": word_preset,
        }
        _write_json(output_dir / "summary.json", summary)
        _write_training_final_results(output_dir, summary=summary)
        heartbeat.update(
            phase="completed",
            message="Custom SAE training complete",
            state="completed",
            activation_count=int(activations.shape[0]),
            checkpoint_path=str(paths["best_model"]),
            layer=layer,
        )
        return summary
    except Exception as error:
        heartbeat.fail(phase=phase, error=error)
        raise


def _encode_vectors(
    *,
    model: SparseAutoencoder,
    vectors: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, vectors.shape[0], batch_size):
            batch = torch.from_numpy(vectors[start : start + batch_size]).to(device).float()
            features = model.encode(batch).detach().cpu().numpy().astype(np.float32)
            batches.append(features)
    return np.vstack(batches)


def _binary_probe_from_features(
    *,
    model: SparseAutoencoder,
    feature_matrix: np.ndarray,
    labels: list[str],
    positive_label: str,
    negative_label: str,
) -> tuple[dict[str, Any] | None, np.ndarray | None]:
    filtered_indices = [
        index
        for index, label in enumerate(labels)
        if label == positive_label or label == negative_label
    ]
    if len(filtered_indices) < 8:
        return None, None
    filtered_labels = [labels[index] for index in filtered_indices]
    counts = Counter(filtered_labels)
    if counts[positive_label] < 4 or counts[negative_label] < 4:
        return None, None
    filtered_features = feature_matrix[np.array(filtered_indices, dtype=np.int64)]
    targets = np.array(
        [1 if label == positive_label else 0 for label in filtered_labels],
        dtype=np.int64,
    )
    split_count = min(5, int(counts[positive_label]), int(counts[negative_label]))
    if split_count < 2:
        return None, None
    clf = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=split_count, shuffle=True, random_state=17)
    scores = cross_val_score(clf, filtered_features, targets, cv=cv, scoring="accuracy")
    clf.fit(filtered_features, targets)
    coefficients = clf.coef_[0].astype(np.float32)
    decoder_vectors = model.decoder_vectors().numpy().astype(np.float32)
    decoder_direction = coefficients @ decoder_vectors
    positive_mean = filtered_features[targets == 1].mean(axis=0)
    negative_mean = filtered_features[targets == 0].mean(axis=0)
    feature_delta = positive_mean - negative_mean
    top_indices = np.argsort(np.abs(coefficients))[::-1][:25]
    summary = {
        "class_counts": {
            negative_label: int(counts[negative_label]),
            positive_label: int(counts[positive_label]),
        },
        "cv_accuracy_mean": float(scores.mean()),
        "cv_accuracy_std": float(scores.std()),
        "negative_label": negative_label,
        "positive_label": positive_label,
        "sample_count": int(filtered_features.shape[0]),
        "top_features": [
            {
                "activation_delta": float(feature_delta[index]),
                "coefficient": float(coefficients[index]),
                "feature": int(index),
                "negative_mean": float(negative_mean[index]),
                "positive_mean": float(positive_mean[index]),
            }
            for index in top_indices
        ],
    }
    return summary, decoder_direction.astype(np.float32)


def _one_vs_rest_family_probe(
    *,
    model: SparseAutoencoder,
    feature_matrix: np.ndarray,
    labels: list[str],
    family: str,
) -> tuple[dict[str, Any] | None, np.ndarray | None]:
    positive_count = sum(1 for label in labels if label == family)
    negative_count = sum(1 for label in labels if label != family)
    if positive_count < 4 or negative_count < 4:
        return None, None
    binary_labels = [family if label == family else "__other__" for label in labels]
    return _binary_probe_from_features(
        model=model,
        feature_matrix=feature_matrix,
        labels=binary_labels,
        positive_label=family,
        negative_label="__other__",
    )


def _sanitize_label_for_filename(label: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return cleaned or "label"


def _load_run_vectors(
    *,
    color_run_dir: Path,
    layer: int,
    format_name: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if format_name == "all":
        all_vectors: list[np.ndarray] = []
        all_rows: list[dict[str, Any]] = []
        for current_format in ("word", "hex", "rgb"):
            vectors, rows = _load_run_vectors(
                color_run_dir=color_run_dir,
                layer=layer,
                format_name=current_format,
            )
            all_vectors.append(vectors)
            all_rows.extend(rows)
        return np.vstack(all_vectors), all_rows
    vectors = np.load(
        color_run_dir / "hidden_states" / format_name / f"layer_{layer:02d}.npy"
    ).astype(np.float32)
    rows = _read_prediction_rows(color_run_dir / f"predictions_{format_name}.jsonl")
    valid_indices = [index for index, row in enumerate(rows) if row.get("color_family") is not None]
    valid_rows = [rows[index] for index in valid_indices]
    return vectors[np.array(valid_indices, dtype=np.int64)], valid_rows


def run_color_sae_feature_analysis(
    *,
    sae_checkpoint_path: Path,
    color_run_dir: Path,
    output_dir: Path,
    layer: int,
    format_name: str = "all",
    batch_size: int = 256,
    device: str = "cpu",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = HeartbeatRecorder(output_dir, label="sae-analyze")
    heartbeat.write_manifest(
        batch_size=batch_size,
        color_run_dir=str(color_run_dir),
        command="sae-analyze",
        device=device,
        format_name=format_name,
        layer=layer,
        sae_checkpoint_path=str(sae_checkpoint_path),
    )
    phase = "setup"
    try:
        model, checkpoint_config = load_sparse_autoencoder_checkpoint(
            sae_checkpoint_path,
            device=device,
        )
        runtime_device = torch.device(device)
        vectors, valid_rows = _load_run_vectors(
            color_run_dir=color_run_dir,
            layer=layer,
            format_name=format_name,
        )
        heartbeat.update(
            phase="encode",
            message="Encoding activations into SAE feature space",
            activation_count=int(vectors.shape[0]),
        )
        encoded_features = _encode_vectors(
            model=model,
            vectors=vectors,
            batch_size=batch_size,
            device=runtime_device,
        )
        np.save(output_dir / "encoded_features.npy", encoded_features.astype(np.float32))

        color_families = [str(row["color_family"]) for row in valid_rows]
        warm_cool_summary, warm_cool_direction = _binary_probe_from_features(
            model=model,
            feature_matrix=encoded_features,
            labels=[str(row["temperature"]) for row in valid_rows],
            positive_label="warm",
            negative_label="cool",
        )
        red_blue_summary, red_blue_direction = _binary_probe_from_features(
            model=model,
            feature_matrix=encoded_features,
            labels=color_families,
            positive_label="red",
            negative_label="blue",
        )
        if warm_cool_direction is not None:
            np.save(output_dir / "warm_cool_decoder_direction.npy", warm_cool_direction)
        if red_blue_direction is not None:
            np.save(output_dir / "red_blue_decoder_direction.npy", red_blue_direction)

        family_rankings: dict[str, Any] = {}
        family_directions_dir = output_dir / "family_decoder_directions"
        family_directions_dir.mkdir(parents=True, exist_ok=True)
        for family, count in sorted(Counter(color_families).items()):
            if count < 4:
                continue
            summary, direction = _one_vs_rest_family_probe(
                model=model,
                feature_matrix=encoded_features,
                labels=color_families,
                family=family,
            )
            if summary is None or direction is None:
                continue
            family_rankings[family] = summary
            np.save(family_directions_dir / f"{family}_decoder_direction.npy", direction)
        _write_json(output_dir / "family_rankings.json", family_rankings)

        schema_label_rankings: dict[str, Any] = {}
        if format_name != "all":
            schema_labels = [str(row["normalized_output"]) for row in valid_rows]
            schema_directions_dir = output_dir / "schema_decoder_directions"
            schema_directions_dir.mkdir(parents=True, exist_ok=True)
            for schema_label, count in sorted(Counter(schema_labels).items()):
                if count < 4:
                    continue
                summary, direction = _one_vs_rest_family_probe(
                    model=model,
                    feature_matrix=encoded_features,
                    labels=schema_labels,
                    family=schema_label,
                )
                if summary is None or direction is None:
                    continue
                schema_label_rankings[schema_label] = summary
                np.save(
                    schema_directions_dir / f"{_sanitize_label_for_filename(schema_label)}_decoder_direction.npy",
                    direction,
                )
        _write_json(output_dir / "schema_label_rankings.json", schema_label_rankings)

        feature_rows = []
        for row_index, row in enumerate(valid_rows):
            feature_rows.append(
                {
                    "active_feature_count": int((encoded_features[row_index] > 0).sum()),
                    "color_family": row["color_family"],
                    "format": row["format"],
                    "normalized_output": row["normalized_output"],
                    "temperature": row["temperature"],
                    "top_features": [
                        int(index)
                        for index in encoded_features[row_index].argsort()[::-1][:10]
                        if encoded_features[row_index, index] > 0
                    ],
                    "word": row["word"],
                }
            )
        _write_jsonl(output_dir / "feature_examples.jsonl", feature_rows)
        summary = {
            "activation_count": int(encoded_features.shape[0]),
            "dictionary_size": int(checkpoint_config["dictionary_size"]),
            "family_rankings": family_rankings,
            "format_name": format_name,
            "generated_at_utc": _utc_now(),
            "layer": layer,
            "red_blue": red_blue_summary,
            "sae_checkpoint_path": str(sae_checkpoint_path),
            "schema_label_rankings": schema_label_rankings,
            "warm_cool": warm_cool_summary,
        }
        _write_json(output_dir / "summary.json", summary)
        _write_analysis_final_results(output_dir, summary=summary)
        heartbeat.update(
            phase="completed",
            message="Custom SAE analysis complete",
            state="completed",
            activation_count=int(encoded_features.shape[0]),
            family_count=len(family_rankings),
            schema_label_count=len(schema_label_rankings),
        )
        return summary
    except Exception as error:
        heartbeat.fail(phase=phase, error=error)
        raise
