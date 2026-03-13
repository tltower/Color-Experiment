from __future__ import annotations

import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .custom_sae import SparseAutoencoder
from .experiment import (
    COLOR_WORD_SYNONYMS,
    FAMILY_PALETTE,
    FORMAT_STROKES,
    FORMAT_PROMPTS,
    HeartbeatRecorder,
    _ensure_checkpoint_state,
    _read_json,
    _read_prediction_rows,
    _save_checkpoint_state,
    _coerce_hidden_output,
    _find_transformer_blocks,
    _move_batch_to_device,
    _non_padding_last_positions,
    _render_prompt,
    _resolve_device,
    _write_json,
    _write_jsonl,
    parse_format_completion,
)
from .hf import create_generation_components
from .word_lists import bundled_color_word_list_path, read_word_file

QWEN_OFF_THE_SHELF_SAE_REPO = "andyrdt/saes-qwen2.5-7b-instruct"
QWEN_OFF_THE_SHELF_SAE_LAYERS: tuple[int, ...] = (3, 7, 11, 15, 19, 23, 27)
TRAINER_INDEX_BY_TOP_K: dict[int, int] = {32: 0, 64: 1, 128: 2, 256: 3}
CORE_COLOR_FAMILIES: tuple[str, ...] = (
    "red",
    "orange",
    "yellow",
    "green",
    "cyan",
    "blue",
    "purple",
    "magenta",
    "brown",
    "black",
    "white",
    "gray",
)
CORE_COLOR_HEX: dict[str, str] = {
    "red": "#ff0000",
    "orange": "#ff8800",
    "yellow": "#ffff00",
    "green": "#00ff00",
    "cyan": "#00ffff",
    "blue": "#0000ff",
    "purple": "#8000ff",
    "magenta": "#ff00ff",
    "brown": "#8b4513",
    "black": "#000000",
    "white": "#ffffff",
    "gray": "#808080",
}
CORE_COLOR_RGB: dict[str, str] = {
    "red": "255,0,0",
    "orange": "255,136,0",
    "yellow": "255,255,0",
    "green": "0,255,0",
    "cyan": "0,255,255",
    "blue": "0,0,255",
    "purple": "128,0,255",
    "magenta": "255,0,255",
    "brown": "139,69,19",
    "black": "0,0,0",
    "white": "255,255,255",
    "gray": "128,128,128",
}
BUILTIN_SEMANTIC_OBJECTS: tuple[str, ...] = (
    "fire",
    "ocean",
    "forest",
    "rose",
    "sun",
    "night",
)


def _require_geometry_stack() -> tuple[Any, Any, Any]:
    try:
        import numpy as np  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        from sklearn.metrics import silhouette_score  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "The research stack is not installed. Use `pip install -e .` in the repo environment."
        ) from exc
    return np, torch, silhouette_score


def _family_for_color_word(word: str) -> str:
    lowered = word.strip().lower()
    return COLOR_WORD_SYNONYMS.get(lowered, lowered)


def _value_schema(value: str) -> str:
    if value.startswith("#"):
        return "hex"
    if "," in value:
        return "rgb"
    return "word"


def _read_color_words(word_list_path: Path | None, limit: int | None) -> list[str]:
    source_path = bundled_color_word_list_path() if word_list_path is None else word_list_path
    words = read_word_file(source_path, limit=limit)
    if not words:
        raise ValueError(f"No valid words loaded from {source_path}")
    return words


def _build_geometry_panel(
    *,
    word_list_path: Path | None,
    include_word_catalog: bool,
    include_anchor_word: bool,
    include_anchor_hex: bool,
    include_anchor_rgb: bool,
    word_limit: int | None,
    prompt_template: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    anchor_count = 0
    if include_anchor_word or include_anchor_hex or include_anchor_rgb:
        for family in CORE_COLOR_FAMILIES:
            values: list[tuple[str, str]] = []
            if include_anchor_word:
                values.append(("word", family))
            if include_anchor_hex:
                values.append(("hex", CORE_COLOR_HEX[family]))
            if include_anchor_rgb:
                values.append(("rgb", CORE_COLOR_RGB[family]))
            for schema, value in values:
                anchor_count += 1
                rows.append(
                    {
                        "color_family": family,
                        "group": "anchor",
                        "prompt": prompt_template.format(value=value),
                        "record_id": f"anchor-{schema}-{family}",
                        "schema": schema,
                        "value": value,
                    }
                )
    catalog_words: list[str] = []
    if include_word_catalog:
        catalog_words = _read_color_words(word_list_path, limit=word_limit)
        for index, word in enumerate(catalog_words):
            rows.append(
                {
                    "color_family": _family_for_color_word(word),
                    "display_label": word,
                    "group": "catalog",
                    "prompt": prompt_template.format(value=word),
                    "record_id": f"catalog-word-{index:04d}",
                    "schema": "word",
                    "value": word,
                }
            )
    if not rows:
        raise ValueError("The geometry panel is empty; enable at least one anchor schema or the word catalog.")
    panel_metadata = {
        "anchor_count": anchor_count,
        "catalog_count": len(catalog_words),
        "prompt_template": prompt_template,
        "word_list_path": None if word_list_path is None else str(word_list_path),
    }
    return rows, panel_metadata


def _off_the_shelf_layer_dir(
    *,
    repo_id_or_path: str,
    layer: int,
    trainer_index: int,
    cache_dir: Path | None,
) -> Path:
    local_root = Path(repo_id_or_path)
    if local_root.exists():
        layer_dir = local_root / f"resid_post_layer_{layer}" / f"trainer_{trainer_index}"
        if not layer_dir.exists():
            raise FileNotFoundError(f"Missing local SAE layer directory: {layer_dir}")
        return layer_dir
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "huggingface_hub is required to download off-the-shelf SAEs. "
            "Install it or pass a local checkpoint directory."
        ) from exc
    filenames = ["ae.pt", "config.json"]
    downloaded: list[Path] = []
    for filename in filenames:
        downloaded.append(
            Path(
                hf_hub_download(
                    repo_id=repo_id_or_path,
                    filename=f"resid_post_layer_{layer}/trainer_{trainer_index}/{filename}",
                    cache_dir=None if cache_dir is None else str(cache_dir),
                )
            )
        )
    return downloaded[0].parent


def _state_dict_from_payload(payload: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        return dict(payload["state_dict"]), dict(payload.get("config", {}))
    if isinstance(payload, dict):
        tensor_like_keys = [key for key, value in payload.items() if hasattr(value, "shape")]
        if tensor_like_keys:
            return dict(payload), {}
    raise RuntimeError("Unsupported SAE checkpoint payload format.")


def _lookup_first(config: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in config:
            return config[key]
    return None


def _infer_sae_dimensions(config: dict[str, Any], state_dict: dict[str, Any]) -> tuple[int, int]:
    input_dim = _lookup_first(config, ("input_dim", "d_in", "activation_dim", "model_dim", "act_size"))
    dictionary_size = _lookup_first(
        config,
        ("dictionary_size", "dict_size", "n_learned_features", "num_features"),
    )
    encoder_weight = state_dict.get("encoder.weight")
    if encoder_weight is None:
        encoder_weight = state_dict.get("W_enc")
    decoder_weight = state_dict.get("decoder.weight")
    if decoder_weight is None:
        decoder_weight = state_dict.get("W_dec")
    if input_dim is None and encoder_weight is not None:
        shape = tuple(int(value) for value in encoder_weight.shape)
        input_dim = shape[1] if shape[0] <= shape[1] else shape[0]
    if dictionary_size is None and encoder_weight is not None:
        shape = tuple(int(value) for value in encoder_weight.shape)
        dictionary_size = shape[0] if shape[0] <= shape[1] else shape[1]
    if input_dim is None and decoder_weight is not None:
        shape = tuple(int(value) for value in decoder_weight.shape)
        input_dim = min(shape)
    if dictionary_size is None and decoder_weight is not None:
        shape = tuple(int(value) for value in decoder_weight.shape)
        dictionary_size = max(shape)
    if input_dim is None or dictionary_size is None:
        raise RuntimeError("Could not infer SAE input_dim / dictionary_size from checkpoint.")
    return int(input_dim), int(dictionary_size)


def _canonicalize_sae_state_dict(
    *,
    torch: Any,
    input_dim: int,
    dictionary_size: int,
    raw_state_dict: dict[str, Any],
) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    if "input_bias" in raw_state_dict:
        canonical["input_bias"] = raw_state_dict["input_bias"]
    elif "b_dec" in raw_state_dict:
        canonical["input_bias"] = raw_state_dict["b_dec"]
    elif "decoder.bias" in raw_state_dict:
        canonical["input_bias"] = raw_state_dict["decoder.bias"]
    else:
        canonical["input_bias"] = torch.zeros(input_dim)

    encoder_weight = raw_state_dict.get("encoder.weight")
    if encoder_weight is None:
        encoder_weight = raw_state_dict.get("W_enc")
    if encoder_weight is None:
        raise RuntimeError("Unsupported SAE checkpoint: missing encoder weight.")
    if tuple(int(value) for value in encoder_weight.shape) == (input_dim, dictionary_size):
        encoder_weight = encoder_weight.T
    canonical["encoder.weight"] = encoder_weight

    encoder_bias = raw_state_dict.get("encoder.bias")
    if encoder_bias is None:
        encoder_bias = raw_state_dict.get("b_enc")
    if encoder_bias is None:
        encoder_bias = torch.zeros(dictionary_size)
    canonical["encoder.bias"] = encoder_bias

    decoder_weight = raw_state_dict.get("decoder.weight")
    if decoder_weight is None:
        decoder_weight = raw_state_dict.get("W_dec")
    if decoder_weight is None:
        raise RuntimeError("Unsupported SAE checkpoint: missing decoder weight.")
    if tuple(int(value) for value in decoder_weight.shape) == (dictionary_size, input_dim):
        decoder_weight = decoder_weight.T
    canonical["decoder.weight"] = decoder_weight
    return canonical


def load_off_the_shelf_sae(
    *,
    layer: int,
    repo_id_or_path: str = QWEN_OFF_THE_SHELF_SAE_REPO,
    trainer_index: int = 0,
    device: str = "cpu",
    cache_dir: Path | None = None,
) -> tuple[SparseAutoencoder, dict[str, Any]]:
    _, torch, _ = _require_geometry_stack()
    layer_dir = _off_the_shelf_layer_dir(
        repo_id_or_path=repo_id_or_path,
        layer=layer,
        trainer_index=trainer_index,
        cache_dir=cache_dir,
    )
    config_path = layer_dir / "config.json"
    checkpoint_path = layer_dir / "ae.pt"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    try:
        payload = torch.load(checkpoint_path, map_location=device)
        raw_state_dict, embedded_config = _state_dict_from_payload(payload)
        if embedded_config:
            merged_config = dict(config)
            merged_config.update(embedded_config)
            config = merged_config
    except Exception as exc:
        try:  # pragma: no cover - optional dependency
            from dictionary_learning import BatchTopKSAE  # type: ignore[import-not-found]
        except Exception as fallback_exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                f"Failed to load off-the-shelf SAE from {checkpoint_path}. "
                "Install `dictionary_learning` or provide a compatible checkpoint."
            ) from (fallback_exc if isinstance(exc, (FileNotFoundError, RuntimeError)) else exc)
        module = BatchTopKSAE.from_pretrained(str(layer_dir))
        state_dict = module.state_dict()
        raw_state_dict = {str(key): value for key, value in state_dict.items()}
        config.setdefault("top_k", _lookup_first(config, ("k", "top_k", "activation_k", "dead_feature_threshold")))
    input_dim, dictionary_size = _infer_sae_dimensions(config, raw_state_dict)
    top_k = _lookup_first(config, ("top_k", "k", "activation_k"))
    canonical_state = _canonicalize_sae_state_dict(
        torch=torch,
        input_dim=input_dim,
        dictionary_size=dictionary_size,
        raw_state_dict=raw_state_dict,
    )
    model = SparseAutoencoder(input_dim=input_dim, dictionary_size=dictionary_size, top_k=top_k)
    model.load_state_dict(canonical_state, strict=False)
    model.to(device)
    model.eval()
    resolved_config = {
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "dictionary_size": dictionary_size,
        "input_dim": input_dim,
        "layer": layer,
        "repo_id_or_path": repo_id_or_path,
        "top_k": top_k,
        "trainer_index": trainer_index,
    }
    resolved_config.update(config)
    return model, resolved_config


def _capture_last_token_activations(
    *,
    model_name: str,
    records: list[dict[str, Any]],
    layers: tuple[int, ...],
    max_length: int,
    batch_size: int,
    device: str,
    heartbeat: HeartbeatRecorder,
    output_dir: Path | None = None,
    resume: bool = False,
    state: dict[str, Any] | None = None,
) -> tuple[dict[int, Any], list[dict[str, Any]]]:
    np, torch, _ = _require_geometry_stack()
    tokenizer, model = create_generation_components(model_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    runtime_device = _resolve_device(torch, device)
    model.to(runtime_device)
    model.eval()
    # Geometry runs intentionally analyze the raw `Color: X` prompt so the last
    # non-padding token stays attached to the color value instead of a
    # chat-template assistant preamble.
    prompts = [str(record["prompt"]) for record in records]
    layer_vectors: dict[int, list[Any]] = {layer: [] for layer in layers}
    enriched_rows: list[dict[str, Any]] = []
    total_batches = math.ceil(len(records) / batch_size)
    for batch_index, start in enumerate(range(0, len(records), batch_size), start=1):
        loaded_checkpoint = (
            _load_geometry_collect_batch_checkpoint(
                np=np,
                output_dir=output_dir,
                batch_index=batch_index,
            )
            if resume and output_dir is not None
            else None
        )
        if loaded_checkpoint is not None:
            batch_rows, batch_layer_arrays = loaded_checkpoint
            enriched_rows.extend(batch_rows)
            for layer, array in batch_layer_arrays.items():
                layer_vectors[layer].append(array.astype(np.float32))
            heartbeat.update(
                phase="collect",
                message=f"Loaded activation batch {batch_index}/{total_batches} from checkpoint",
                batch_index=batch_index,
                total_batches=total_batches,
                processed_records=min(start + batch_size, len(records)),
                total_records=len(records),
            )
            if state is not None and output_dir is not None:
                completed_batches = set(int(value) for value in state.get("completed_collect_batches", []))
                completed_batches.add(batch_index)
                state["completed_collect_batches"] = sorted(completed_batches)
                _save_checkpoint_state(output_dir, "sae_geometry", state)
            continue
        batch_prompts = prompts[start : start + batch_size]
        batch_records = records[start : start + batch_size]
        tokenized = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokenized_device = _move_batch_to_device(tokenized, runtime_device)
        last_positions = _non_padding_last_positions(tokenized_device["attention_mask"])
        heartbeat.update(
            phase="collect",
            message=f"Collecting activations batch {batch_index}/{total_batches}",
            batch_index=batch_index,
            total_batches=total_batches,
            processed_records=min(start + batch_size, len(records)),
            total_records=len(records),
        )
        with torch.no_grad():
            outputs = model(**tokenized_device)
        hidden_states = outputs.hidden_states
        for layer in layers:
            hidden_index = layer + 1
            if hidden_index >= len(hidden_states):
                raise ValueError(
                    f"Requested SAE layer {layer}, but model only returned hidden states up to {len(hidden_states) - 2}."
                )
            last_hidden = hidden_states[hidden_index]
            for row_index, last_position in enumerate(last_positions):
                layer_vectors[layer].append(
                    last_hidden[row_index, last_position, :].detach().float().cpu().numpy().astype(np.float32)
                )
        input_ids = tokenized_device["input_ids"].detach().cpu()
        for row_index, record in enumerate(batch_records):
            last_token_id = int(input_ids[row_index, last_positions[row_index]].item())
            decode = getattr(tokenizer, "decode", None)
            token_text = str(last_token_id)
            if callable(decode):
                try:
                    token_text = str(
                        decode(
                            [last_token_id],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                    )
                except TypeError:
                    token_text = str(decode([last_token_id]))
            enriched = dict(record)
            enriched["last_token_id"] = last_token_id
            enriched["last_token_text"] = token_text
            enriched_rows.append(enriched)
        if output_dir is not None:
            batch_layer_arrays = {
                layer: np.stack(layer_vectors[layer][-len(batch_records) :]).astype(np.float32)
                for layer in layers
            }
            _save_geometry_collect_batch_checkpoint(
                np=np,
                output_dir=output_dir,
                batch_index=batch_index,
                rows=[dict(row) for row in enriched_rows[-len(batch_records) :]],
                layer_arrays=batch_layer_arrays,
            )
        if state is not None and output_dir is not None:
            completed_batches = set(int(value) for value in state.get("completed_collect_batches", []))
            completed_batches.add(batch_index)
            state["completed_collect_batches"] = sorted(completed_batches)
            _save_checkpoint_state(output_dir, "sae_geometry", state)
    return {layer: np.vstack(vectors) for layer, vectors in layer_vectors.items()}, enriched_rows


def _encode_activations(
    *,
    torch: Any,
    sae: SparseAutoencoder,
    activations: Any,
    batch_size: int,
    device: str,
) -> Any:
    encoded_batches = []
    with torch.no_grad():
        for start in range(0, int(activations.shape[0]), batch_size):
            batch = torch.from_numpy(activations[start : start + batch_size]).to(device).float()
            encoded = sae.encode(batch).detach().cpu().numpy()
            encoded_batches.append(encoded)
    return __import__("numpy").vstack(encoded_batches).astype(__import__("numpy").float32)


def _geometry_collect_checkpoint_dir(output_dir: Path) -> Path:
    return output_dir / "checkpoints" / "collect_batches"


def _geometry_collect_batch_rows_path(output_dir: Path, batch_index: int) -> Path:
    return _geometry_collect_checkpoint_dir(output_dir) / f"batch_{batch_index:04d}.rows.jsonl"


def _geometry_collect_batch_hidden_path(output_dir: Path, batch_index: int) -> Path:
    return _geometry_collect_checkpoint_dir(output_dir) / f"batch_{batch_index:04d}.activations.npz"


def _save_geometry_collect_batch_checkpoint(
    *,
    np: Any,
    output_dir: Path,
    batch_index: int,
    rows: list[dict[str, Any]],
    layer_arrays: dict[int, Any],
) -> None:
    checkpoint_dir = _geometry_collect_checkpoint_dir(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(_geometry_collect_batch_rows_path(output_dir, batch_index), rows)
    np.savez_compressed(
        _geometry_collect_batch_hidden_path(output_dir, batch_index),
        **{f"layer_{layer:02d}": array.astype(np.float16) for layer, array in layer_arrays.items()},
    )


def _load_geometry_collect_batch_checkpoint(
    *,
    np: Any,
    output_dir: Path | None,
    batch_index: int,
) -> tuple[list[dict[str, Any]], dict[int, Any]] | None:
    if output_dir is None:
        return None
    rows_path = _geometry_collect_batch_rows_path(output_dir, batch_index)
    hidden_path = _geometry_collect_batch_hidden_path(output_dir, batch_index)
    if not rows_path.exists() or not hidden_path.exists():
        return None
    rows = _read_prediction_rows(rows_path)
    layer_arrays: dict[int, Any] = {}
    with np.load(hidden_path) as payload:
        for key in payload.files:
            layer_arrays[int(key.rsplit("_", 1)[1])] = payload[key].astype(np.float32)
    return rows, layer_arrays


def _geometry_activation_path(output_dir: Path, layer: int) -> Path:
    return output_dir / "activations" / f"layer_{layer:02d}.npy"


def _load_geometry_capture(
    *,
    np: Any,
    output_dir: Path,
    layers: tuple[int, ...],
) -> tuple[dict[int, Any], list[dict[str, Any]]] | None:
    panel_path = output_dir / "panel.jsonl"
    if not panel_path.exists():
        return None
    activation_paths = [_geometry_activation_path(output_dir, layer) for layer in layers]
    if not all(path.exists() for path in activation_paths):
        return None
    rows = _read_prediction_rows(panel_path)
    activations = {
        layer: np.load(_geometry_activation_path(output_dir, layer)).astype(np.float32)
        for layer in layers
    }
    return activations, rows


def _eta_squared(values: Any, labels: list[str], *, np: Any) -> Any:
    if values.shape[0] == 0:
        return np.zeros(values.shape[1], dtype=np.float32)
    total_mean = values.mean(axis=0)
    total_ss = ((values - total_mean) ** 2).sum(axis=0)
    if np.all(total_ss <= 1e-8):
        return np.zeros(values.shape[1], dtype=np.float32)
    between_ss = np.zeros(values.shape[1], dtype=np.float32)
    label_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        label_to_indices[str(label)].append(index)
    for indices in label_to_indices.values():
        group = values[np.array(indices, dtype=np.int64)]
        if group.shape[0] == 0:
            continue
        delta = group.mean(axis=0) - total_mean
        between_ss += float(group.shape[0]) * (delta**2)
    return np.divide(
        between_ss,
        total_ss + 1e-8,
        out=np.zeros_like(between_ss),
        where=(total_ss > 1e-8),
    )


def _silhouette_or_none(*, values: Any, labels: list[str], silhouette_score: Any) -> float | None:
    if values.shape[0] < 3 or len(set(labels)) < 2:
        return None
    counts = Counter(labels)
    if min(counts.values()) < 2:
        return None
    return float(silhouette_score(values, labels))


def _torch_pca_projection(
    *,
    torch: Any,
    values: Any,
    device: str,
) -> tuple[Any, tuple[float, float]]:
    np = __import__("numpy")

    def _project(target_device: str) -> tuple[Any, tuple[float, float]]:
        tensor = torch.from_numpy(values).to(target_device).float()
        row_count = int(tensor.shape[0])
        if row_count == 0:
            return np.zeros((0, 2), dtype=np.float32), (0.0, 0.0)
        if row_count == 1 or int(tensor.shape[1]) == 0:
            return (
                np.zeros((row_count, 2), dtype=np.float32),
                (0.0, 0.0),
            )
        q = max(1, min(2, row_count, int(tensor.shape[1])))
        centered = tensor - tensor.mean(dim=0, keepdim=True)
        total_variance = float((centered.pow(2).sum() / max(row_count - 1, 1)).item())
        u, singular_values, _v = torch.pca_lowrank(tensor, q=q, center=True)
        coords = u[:, :q] * singular_values[:q]
        coords_2d = torch.zeros((row_count, 2), device=tensor.device, dtype=tensor.dtype)
        coords_2d[:, :q] = coords
        explained = (singular_values[:q] ** 2) / max(row_count - 1, 1)
        ratios = torch.zeros(2, device=tensor.device, dtype=tensor.dtype)
        if total_variance > 1e-8:
            ratios[:q] = explained / total_variance
        return coords_2d.detach().cpu().numpy().astype(np.float32), (
            float(ratios[0].item()),
            float(ratios[1].item()),
        )

    try:
        return _project(device)
    except RuntimeError:
        if str(device) == "cpu":
            raise
        return _project("cpu")


def _centroid_accuracy(
    *,
    np: Any,
    values: Any,
    rows: list[dict[str, Any]],
    source_schema: str,
    target_schemas: tuple[str, ...],
) -> float | None:
    source_indices = [
        index for index, row in enumerate(rows) if row["group"] == "anchor" and row["schema"] == source_schema
    ]
    target_indices = [
        index
        for index, row in enumerate(rows)
        if row["group"] == "anchor" and row["schema"] in target_schemas
    ]
    if not source_indices or not target_indices:
        return None
    centroids: dict[str, Any] = {}
    for family in CORE_COLOR_FAMILIES:
        family_indices = [
            index
            for index in source_indices
            if str(rows[index].get("color_family")) == family
        ]
        if family_indices:
            centroids[family] = values[np.array(family_indices, dtype=np.int64)].mean(axis=0)
    if len(centroids) < 2:
        return None
    correct = 0
    for index in target_indices:
        vector = values[index]
        predicted_family = min(
            centroids,
            key=lambda family: float(((vector - centroids[family]) ** 2).sum()),
        )
        if predicted_family == rows[index]["color_family"]:
            correct += 1
    return float(correct / len(target_indices))


def _top_rankings(
    *,
    scores: Any,
    labels: list[str],
    top_n: int,
) -> list[dict[str, Any]]:
    rows = [
        {"feature": int(index), "label": labels[index], "score": float(scores[index])}
        for index in range(len(labels))
    ]
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows[:top_n]


def _top_family_feature_rankings(
    *,
    np: Any,
    encoded: Any,
    rows: list[dict[str, Any]],
    top_n: int,
) -> dict[str, list[dict[str, Any]]]:
    rankings: dict[str, list[dict[str, Any]]] = {}
    for family in CORE_COLOR_FAMILIES:
        positive_indices = [
            index for index, row in enumerate(rows) if row["group"] == "anchor" and row["color_family"] == family
        ]
        negative_indices = [
            index for index, row in enumerate(rows) if row["group"] == "anchor" and row["color_family"] != family
        ]
        if not positive_indices or not negative_indices:
            rankings[family] = []
            continue
        positive_mean = encoded[np.array(positive_indices, dtype=np.int64)].mean(axis=0)
        negative_mean = encoded[np.array(negative_indices, dtype=np.int64)].mean(axis=0)
        delta = positive_mean - negative_mean
        feature_rows = [
            {
                "delta": float(delta[index]),
                "feature": int(index),
                "negative_mean": float(negative_mean[index]),
                "positive_mean": float(positive_mean[index]),
            }
            for index in range(int(encoded.shape[1]))
        ]
        feature_rows.sort(key=lambda row: abs(row["delta"]), reverse=True)
        rankings[family] = feature_rows[:top_n]
    return rankings


def _warm_cool_direction(
    *,
    np: Any,
    encoded: Any,
    rows: list[dict[str, Any]],
    decoder_vectors: Any,
) -> Any | None:
    warm_families = {"red", "orange", "yellow", "brown", "magenta"}
    cool_families = {"green", "cyan", "blue", "purple"}
    warm_indices = [
        index for index, row in enumerate(rows) if row["group"] == "anchor" and row["color_family"] in warm_families
    ]
    cool_indices = [
        index for index, row in enumerate(rows) if row["group"] == "anchor" and row["color_family"] in cool_families
    ]
    if not warm_indices or not cool_indices:
        return None
    delta = encoded[np.array(warm_indices, dtype=np.int64)].mean(axis=0) - encoded[
        np.array(cool_indices, dtype=np.int64)
    ].mean(axis=0)
    return np.matmul(delta.astype(np.float32), decoder_vectors.astype(np.float32)).astype(np.float32)


def _pairwise_direction(
    *,
    np: Any,
    encoded: Any,
    rows: list[dict[str, Any]],
    decoder_vectors: Any,
    positive_family: str,
    negative_family: str,
) -> Any | None:
    positive_indices = [
        index
        for index, row in enumerate(rows)
        if row["group"] == "anchor" and row["color_family"] == positive_family
    ]
    negative_indices = [
        index
        for index, row in enumerate(rows)
        if row["group"] == "anchor" and row["color_family"] == negative_family
    ]
    if not positive_indices or not negative_indices:
        return None
    delta = encoded[np.array(positive_indices, dtype=np.int64)].mean(axis=0) - encoded[
        np.array(negative_indices, dtype=np.int64)
    ].mean(axis=0)
    return np.matmul(delta.astype(np.float32), decoder_vectors.astype(np.float32)).astype(np.float32)


def _projection_bounds(coords: Any) -> tuple[float, float, float, float]:
    min_x = float(coords[:, 0].min())
    max_x = float(coords[:, 0].max())
    min_y = float(coords[:, 1].min())
    max_y = float(coords[:, 1].max())
    if math.isclose(min_x, max_x):
        max_x = min_x + 1.0
    if math.isclose(min_y, max_y):
        max_y = min_y + 1.0
    return min_x, max_x, min_y, max_y


def _marker_shape(*, schema: str, x: float, y: float, fill: str, stroke: str, title: str) -> str:
    if schema == "word":
        return (
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="{fill}" stroke="{stroke}" stroke-width="0.9">'
            f"<title>{html.escape(title)}</title></circle>"
        )
    if schema == "hex":
        return (
            f'<rect x="{x - 4.2:.2f}" y="{y - 4.2:.2f}" width="8.4" height="8.4" rx="1.8" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="0.9"><title>{html.escape(title)}</title></rect>'
        )
    return (
        f'<polygon points="{x:.2f},{y - 4.8:.2f} {x + 4.8:.2f},{y + 4.0:.2f} {x - 4.8:.2f},{y + 4.0:.2f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="0.9"><title>{html.escape(title)}</title></polygon>'
    )


def _write_encoded_pca_svg(
    path: Path,
    *,
    coords: Any,
    rows: list[dict[str, Any]],
    layer: int,
    pc1_variance: float,
    pc2_variance: float,
) -> None:
    width = 960.0
    height = 720.0
    margin = 72.0
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
        f'<text x="{margin}" y="34" font-family="Helvetica, Arial, sans-serif" font-size="24" fill="#111111">SAE geometry PCA layer {layer}</text>',
        f'<text x="{margin}" y="58" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#555555">PC1 {pc1_variance:.1%} | PC2 {pc2_variance:.1%} | points={len(rows)}</text>',
        f'<rect x="{margin}" y="{margin}" width="{inner_width}" height="{inner_height}" fill="none" stroke="#bbb5ab" stroke-width="1" />',
    ]
    for row, coord in zip(rows, coords.tolist(), strict=True):
        fill = FAMILY_PALETTE.get(str(row.get("color_family")), "#777777")
        schema = str(row["schema"])
        title = f'{row["record_id"]} | schema={schema} | family={row.get("color_family")} | value={row.get("value")}'
        lines.append(
            _marker_shape(
                schema=schema,
                x=project_x(float(coord[0])),
                y=project_y(float(coord[1])),
                fill=fill,
                stroke=FORMAT_STROKES.get(schema, "#222222"),
                title=title,
            )
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _normalize_direction(np: Any, direction: Any) -> Any:
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        return direction.astype(np.float32)
    return (direction / norm).astype(np.float32)


def _analyze_layer(
    *,
    output_dir: Path,
    layer: int,
    rows: list[dict[str, Any]],
    activations: Any,
    sae: SparseAutoencoder,
    sae_config: dict[str, Any],
    batch_size: int,
    device: str,
    heartbeat: HeartbeatRecorder,
    np: Any,
    torch: Any,
    silhouette_score: Any,
    compute_silhouette: bool,
) -> dict[str, Any]:
    heartbeat.update(
        phase="analyze",
        message=f"Analyzing SAE geometry at layer {layer}",
        current_layer=layer,
    )
    encoded = _encode_activations(
        torch=__import__("torch"),
        sae=sae,
        activations=activations,
        batch_size=batch_size,
        device=device,
    )
    layer_dir = output_dir / f"layer_{layer:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    np.save(layer_dir / "encoded_features.npy", encoded.astype(np.float32))
    decoder_vectors = sae.decoder_vectors().detach().cpu().numpy().astype(np.float32)
    anchor_indices = [index for index, row in enumerate(rows) if row["group"] == "anchor"]
    anchor_rows = [rows[index] for index in anchor_indices]
    anchor_encoded = encoded[np.array(anchor_indices, dtype=np.int64)] if anchor_indices else encoded[:0]
    family_labels = [str(row["color_family"]) for row in anchor_rows]
    schema_labels = [str(row["schema"]) for row in anchor_rows]
    color_scores = _eta_squared(anchor_encoded, family_labels, np=np) if anchor_indices else np.zeros(encoded.shape[1], dtype=np.float32)
    format_scores = _eta_squared(anchor_encoded, schema_labels, np=np) if anchor_indices else np.zeros(encoded.shape[1], dtype=np.float32)
    total_variance = encoded.var(axis=0).astype(np.float32)
    invariant_scores = color_scores - format_scores
    feature_rows = [
        {
            "color_eta_squared": float(color_scores[index]),
            "feature": int(index),
            "format_eta_squared": float(format_scores[index]),
            "invariant_score": float(invariant_scores[index]),
            "total_variance": float(total_variance[index]),
        }
        for index in range(int(encoded.shape[1]))
    ]
    feature_rows.sort(key=lambda row: row["invariant_score"], reverse=True)
    _write_jsonl(layer_dir / "feature_scores.jsonl", feature_rows)
    family_rankings = _top_family_feature_rankings(np=np, encoded=encoded, rows=rows, top_n=24)
    _write_json(layer_dir / "family_feature_rankings.json", family_rankings)
    color_score_rows = sorted(
        feature_rows,
        key=lambda row: row["color_eta_squared"],
        reverse=True,
    )[:24]
    format_score_rows = sorted(
        feature_rows,
        key=lambda row: row["format_eta_squared"],
        reverse=True,
    )[:24]
    _write_json(layer_dir / "top_invariant_features.json", {"features": feature_rows[:24]})
    _write_json(layer_dir / "top_color_features.json", {"features": color_score_rows})
    _write_json(layer_dir / "top_format_features.json", {"features": format_score_rows})
    directions_dir = layer_dir / "directions"
    directions_dir.mkdir(parents=True, exist_ok=True)
    for family in CORE_COLOR_FAMILIES:
        ranking = family_rankings.get(family, [])
        if not ranking:
            continue
        delta = np.zeros(encoded.shape[1], dtype=np.float32)
        for row in ranking:
            delta[int(row["feature"])] = float(row["delta"])
        direction = np.matmul(delta, decoder_vectors).astype(np.float32)
        np.save(directions_dir / f"{family}_direction.npy", _normalize_direction(np, direction))
    red_blue_direction = _pairwise_direction(
        np=np,
        encoded=encoded,
        rows=rows,
        decoder_vectors=decoder_vectors,
        positive_family="red",
        negative_family="blue",
    )
    if red_blue_direction is not None:
        np.save(directions_dir / "red_blue_direction.npy", _normalize_direction(np, red_blue_direction))
    warm_cool_direction = _warm_cool_direction(
        np=np,
        encoded=encoded,
        rows=rows,
        decoder_vectors=decoder_vectors,
    )
    if warm_cool_direction is not None:
        np.save(directions_dir / "warm_cool_direction.npy", _normalize_direction(np, warm_cool_direction))
    coords, (pc1_variance, pc2_variance) = _torch_pca_projection(
        torch=torch,
        values=anchor_encoded if anchor_indices else encoded,
        device=device,
    )
    pca_rows = anchor_rows if anchor_indices else rows
    _write_encoded_pca_svg(
        layer_dir / "encoded_pca.svg",
        coords=coords,
        rows=pca_rows,
        layer=layer,
        pc1_variance=pc1_variance,
        pc2_variance=pc2_variance,
    )
    pca_rows_json = [
        {
            "color_family": row.get("color_family"),
            "group": row.get("group"),
            "pc1": float(coord[0]),
            "pc2": float(coord[1]),
            "record_id": row["record_id"],
            "schema": row["schema"],
            "value": row.get("value"),
        }
        for row, coord in zip(pca_rows, coords.tolist(), strict=True)
    ]
    _write_jsonl(layer_dir / "encoded_pca_points.jsonl", pca_rows_json)
    family_silhouette = (
        _silhouette_or_none(values=anchor_encoded, labels=family_labels, silhouette_score=silhouette_score)
        if compute_silhouette
        else None
    )
    schema_silhouette = (
        _silhouette_or_none(values=anchor_encoded, labels=schema_labels, silhouette_score=silhouette_score)
        if compute_silhouette
        else None
    )
    word_anchor_transfer = _centroid_accuracy(
        np=np,
        values=encoded,
        rows=rows,
        source_schema="word",
        target_schemas=("hex", "rgb"),
    )
    top_invariant_score = None if not feature_rows else float(feature_rows[0]["invariant_score"])
    summary = {
        "anchor_count": len(anchor_rows),
        "dictionary_size": int(encoded.shape[1]),
        "family_silhouette": family_silhouette,
        "format_silhouette": schema_silhouette,
        "input_dim": int(activations.shape[1]),
        "layer": layer,
        "sae_checkpoint": sae_config.get("checkpoint_path"),
        "sae_repo_id_or_path": sae_config.get("repo_id_or_path"),
        "top_color_feature": None if not color_score_rows else int(color_score_rows[0]["feature"]),
        "top_format_feature": None if not format_score_rows else int(format_score_rows[0]["feature"]),
        "top_invariant_feature": None if not feature_rows else int(feature_rows[0]["feature"]),
        "top_invariant_score": top_invariant_score,
        "word_anchor_transfer_accuracy": word_anchor_transfer,
    }
    _write_json(layer_dir / "summary.json", summary)
    return summary


def _write_geometry_report(path: Path, *, summary: dict[str, Any], layers: list[dict[str, Any]]) -> None:
    best_transfer_layer = summary.get("best_transfer_layer")
    best_invariant_layer = summary.get("best_invariant_layer")
    lines = [
        "# SAE color geometry report",
        "",
        f"- Model: `{summary['model_name']}`",
        f"- Prompt template: `{summary['prompt_template']}`",
        f"- Records: `{summary['record_count']}`",
        f"- Anchor records: `{summary['anchor_count']}`",
        f"- Catalog records: `{summary['catalog_count']}`",
        f"- SAE repo/path: `{summary['sae_repo_id_or_path']}`",
        f"- Layers analyzed: `{summary['layers']}`",
        f"- Best transfer layer: `{best_transfer_layer}`",
        f"- Best invariant layer: `{best_invariant_layer}`",
        "",
        "Per-layer summary:",
        "",
    ]
    for row in layers:
        transfer = row.get("word_anchor_transfer_accuracy")
        transfer_text = "n/a" if transfer is None else f"{transfer:.3f}"
        invariant = row.get("top_invariant_feature")
        lines.append(
            f"- Layer `{row['layer']}`: transfer `{transfer_text}`, family silhouette `{row.get('family_silhouette')}`, invariant feature `{invariant}`"
        )
    lines.extend(
        [
            "",
            "Artifacts to inspect first:",
            "",
            "- `layer_summary.jsonl`",
            "- `summary.json`",
            "- `layer_XX/encoded_pca.svg`",
            "- `layer_XX/top_invariant_features.json`",
            "- `layer_XX/family_feature_rankings.json`",
            "- `layer_XX/directions/`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_geometry_final_results(output_dir: Path, *, summary: dict[str, Any]) -> None:
    payload = {
        "kind": "sae_geometry",
        "key_artifacts": {
            "layer_summary": "layer_summary.jsonl",
            "report": "report.md",
            "summary": "summary.json",
        },
        "summary": summary,
    }
    _write_json(output_dir / "final_results.json", payload)


def run_color_sae_geometry_experiment(
    *,
    output_dir: Path,
    model_name: str,
    sae_repo_id_or_path: str = QWEN_OFF_THE_SHELF_SAE_REPO,
    sae_layers: tuple[int, ...] = QWEN_OFF_THE_SHELF_SAE_LAYERS,
    trainer_index: int = 0,
    cache_dir: Path | None = None,
    word_list_path: Path | None = None,
    prompt_template: str = "Color: {value}",
    include_word_catalog: bool = True,
    include_anchor_word: bool = True,
    include_anchor_hex: bool = True,
    include_anchor_rgb: bool = True,
    word_limit: int | None = None,
    batch_size: int = 64,
    max_length: int = 64,
    encode_batch_size: int = 256,
    device: str = "auto",
    resume: bool = False,
    compute_silhouette: bool = True,
) -> dict[str, Any]:
    np, torch, silhouette_score = _require_geometry_stack()
    output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = HeartbeatRecorder(output_dir, label="sae-geometry")
    panel_rows, panel_metadata = _build_geometry_panel(
        word_list_path=word_list_path,
        include_word_catalog=include_word_catalog,
        include_anchor_word=include_anchor_word,
        include_anchor_hex=include_anchor_hex,
        include_anchor_rgb=include_anchor_rgb,
        word_limit=word_limit,
        prompt_template=prompt_template,
    )
    layers = tuple(int(layer) for layer in sae_layers)
    config = {
        "batch_size": batch_size,
        "cache_dir": None if cache_dir is None else str(cache_dir),
        "compute_silhouette": compute_silhouette,
        "device": device,
        "encode_batch_size": encode_batch_size,
        "include_anchor_hex": include_anchor_hex,
        "include_anchor_rgb": include_anchor_rgb,
        "include_anchor_word": include_anchor_word,
        "include_word_catalog": include_word_catalog,
        "max_length": max_length,
        "model_name": model_name,
        "prompt_template": prompt_template,
        "resume_version": 1,
        "sae_layers": list(layers),
        "sae_repo_id_or_path": sae_repo_id_or_path,
        "trainer_index": trainer_index,
        "word_limit": word_limit,
        "word_list_path": None if word_list_path is None else str(word_list_path),
    }
    state = _ensure_checkpoint_state(
        output_dir=output_dir,
        name="sae_geometry",
        config=config,
        resume=resume,
    )
    state.setdefault("capture_complete", False)
    state.setdefault("completed_collect_batches", [])
    state.setdefault("completed_layers", [])
    heartbeat.write_manifest(
        command="sae-geometry",
        model_name=model_name,
        record_count=len(panel_rows),
        resume=resume,
        sae_layers=list(layers),
        sae_repo_id_or_path=sae_repo_id_or_path,
    )
    runtime_device = _resolve_device(torch, device)
    cached_capture = _load_geometry_capture(np=np, output_dir=output_dir, layers=layers) if resume else None
    if cached_capture is not None:
        activations_by_layer, panel_rows = cached_capture
        state["capture_complete"] = True
        _save_checkpoint_state(output_dir, "sae_geometry", state)
        heartbeat.update(
            phase="collect",
            message="Loaded completed activation capture from checkpoint",
            processed_records=len(panel_rows),
            total_records=len(panel_rows),
        )
    else:
        activations_by_layer, panel_rows = _capture_last_token_activations(
            model_name=model_name,
            records=panel_rows,
            layers=layers,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
            heartbeat=heartbeat,
            output_dir=output_dir,
            resume=resume,
            state=state,
        )
        _write_jsonl(output_dir / "panel.jsonl", panel_rows)
        activations_dir = output_dir / "activations"
        activations_dir.mkdir(parents=True, exist_ok=True)
        for layer, activations in activations_by_layer.items():
            np.save(_geometry_activation_path(output_dir, layer), activations.astype(np.float16))
        state["capture_complete"] = True
        _save_checkpoint_state(output_dir, "sae_geometry", state)
    layer_summaries: list[dict[str, Any]] = []
    for layer in layers:
        summary_path = output_dir / f"layer_{layer:02d}" / "summary.json"
        if resume and summary_path.exists():
            layer_summary = _read_json(summary_path)
            layer_summaries.append(layer_summary)
            completed_layers = set(int(value) for value in state.get("completed_layers", []))
            completed_layers.add(layer)
            state["completed_layers"] = sorted(completed_layers)
            _save_checkpoint_state(output_dir, "sae_geometry", state)
            heartbeat.update(
                phase="analyze",
                message=f"Loaded analyzed layer {layer} from checkpoint",
                current_layer=layer,
            )
            continue
        sae, sae_config = load_off_the_shelf_sae(
            layer=layer,
            repo_id_or_path=sae_repo_id_or_path,
            trainer_index=trainer_index,
            device=str(runtime_device),
            cache_dir=cache_dir,
        )
        layer_summary = _analyze_layer(
            output_dir=output_dir,
            layer=layer,
            rows=panel_rows,
            activations=activations_by_layer[layer],
            sae=sae,
            sae_config=sae_config,
            batch_size=encode_batch_size,
            device=str(runtime_device),
            heartbeat=heartbeat,
            np=np,
            torch=torch,
            silhouette_score=silhouette_score,
            compute_silhouette=compute_silhouette,
        )
        layer_summaries.append(layer_summary)
        completed_layers = set(int(value) for value in state.get("completed_layers", []))
        completed_layers.add(layer)
        state["completed_layers"] = sorted(completed_layers)
        _save_checkpoint_state(output_dir, "sae_geometry", state)
    layer_summaries.sort(key=lambda row: int(row["layer"]))
    _write_jsonl(output_dir / "layer_summary.jsonl", layer_summaries)
    best_transfer_row = max(
        (row for row in layer_summaries if row.get("word_anchor_transfer_accuracy") is not None),
        key=lambda row: float(row["word_anchor_transfer_accuracy"]),
        default=None,
    )
    best_invariant_row = max(
        (row for row in layer_summaries if row.get("top_invariant_score") is not None),
        key=lambda row: float(row["top_invariant_score"]),
        default=None,
    )
    summary = {
        "anchor_count": int(panel_metadata["anchor_count"]),
        "best_invariant_layer": None if best_invariant_row is None else int(best_invariant_row["layer"]),
        "best_transfer_accuracy": None
        if best_transfer_row is None
        else float(best_transfer_row["word_anchor_transfer_accuracy"]),
        "best_transfer_layer": None if best_transfer_row is None else int(best_transfer_row["layer"]),
        "catalog_count": int(panel_metadata["catalog_count"]),
        "layers": list(layers),
        "model_name": model_name,
        "prompt_template": prompt_template,
        "record_count": len(panel_rows),
        "sae_repo_id_or_path": sae_repo_id_or_path,
        "trainer_index": trainer_index,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_geometry_report(output_dir / "report.md", summary=summary, layers=layer_summaries)
    _write_geometry_final_results(output_dir, summary=summary)
    heartbeat.update(phase="complete", message="SAE geometry experiment complete", state="completed")
    return summary


def _intervention_prompts(
    *,
    prompt_mode: str,
    prompt_file: Path | None,
) -> tuple[list[dict[str, Any]], str]:
    if prompt_file is not None:
        rows = []
        for index, raw_line in enumerate(prompt_file.read_text(encoding="utf-8").splitlines()):
            line = raw_line.strip()
            if not line:
                continue
            rows.append({"prompt_id": f"file-{index:04d}", "prompt": line})
        if not rows:
            raise ValueError(f"No prompts loaded from {prompt_file}")
        return rows, "hex"
    if prompt_mode == "blank_hex":
        return [{"prompt_id": "blank-hex", "prompt": "Hex code for:"}], "hex"
    if prompt_mode == "semantic_hex":
        return (
            [
                {
                    "prompt_id": f"semantic-{word}",
                    "prompt": FORMAT_PROMPTS["hex"].format(word=word),
                }
                for word in BUILTIN_SEMANTIC_OBJECTS
            ],
            "hex",
        )
    raise ValueError(f"Unsupported intervention prompt mode {prompt_mode!r}")


def _parse_alphas(alpha_values: str) -> list[float]:
    values = [float(chunk.strip()) for chunk in alpha_values.split(",") if chunk.strip()]
    if not values:
        raise ValueError("At least one intervention alpha is required.")
    return values


def _direction_path(geometry_dir: Path, layer: int, family: str) -> Path:
    return geometry_dir / f"layer_{layer:02d}" / "directions" / f"{family}_direction.npy"


def _load_direction(np: Any, geometry_dir: Path, layer: int, family: str) -> Any:
    path = _direction_path(geometry_dir, layer, family)
    if not path.exists():
        raise FileNotFoundError(f"Missing direction for family {family!r} at layer {layer}: {path}")
    return np.load(path).astype(np.float32)


def _write_intervention_report(path: Path, *, summary: dict[str, Any]) -> None:
    lines = [
        "# SAE direction intervention report",
        "",
        f"- Model: `{summary['model_name']}`",
        f"- Target family: `{summary['target_family']}`",
        f"- Layer: `{summary['layer']}`",
        f"- Prompt mode: `{summary['prompt_mode']}`",
        f"- Alphas: `{summary['alphas']}`",
        f"- Best alpha: `{summary['best_alpha']}`",
        f"- Best target-family match rate: `{summary['best_target_match_rate']}`",
        "",
        "Artifacts to inspect first:",
        "",
        "- `intervention_rows.jsonl`",
        "- `alpha_summary.jsonl`",
        "- `summary.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_intervention_final_results(output_dir: Path, *, summary: dict[str, Any]) -> None:
    payload = {
        "kind": "sae_direction_intervention",
        "key_artifacts": {
            "alpha_summary": "alpha_summary.jsonl",
            "intervention_rows": "intervention_rows.jsonl",
            "report": "report.md",
            "summary": "summary.json",
        },
        "summary": summary,
    }
    _write_json(output_dir / "final_results.json", payload)


def _intervention_checkpoint_dir(output_dir: Path) -> Path:
    return output_dir / "checkpoints" / "intervention_batches"


def _intervention_batch_path(output_dir: Path, batch_index: int) -> Path:
    return _intervention_checkpoint_dir(output_dir) / f"batch_{batch_index:04d}.jsonl"


def _save_intervention_batch_checkpoint(output_dir: Path, batch_index: int, rows: list[dict[str, Any]]) -> None:
    checkpoint_dir = _intervention_checkpoint_dir(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(_intervention_batch_path(output_dir, batch_index), rows)


def _load_intervention_batch_checkpoint(output_dir: Path, batch_index: int) -> list[dict[str, Any]] | None:
    path = _intervention_batch_path(output_dir, batch_index)
    if not path.exists():
        return None
    return _read_prediction_rows(path)


def run_color_direction_intervention_experiment(
    *,
    output_dir: Path,
    geometry_dir: Path,
    model_name: str,
    layer: int,
    family: str,
    alpha_values: str = "-8,-4,-2,-1,0,1,2,4,8",
    prompt_mode: str = "blank_hex",
    prompt_file: Path | None = None,
    batch_size: int = 8,
    max_length: int = 128,
    max_new_tokens: int = 16,
    device: str = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    np, torch, _ = _require_geometry_stack()
    output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = HeartbeatRecorder(output_dir, label="sae-intervene")
    prompts, parse_format = _intervention_prompts(prompt_mode=prompt_mode, prompt_file=prompt_file)
    alphas = _parse_alphas(alpha_values)
    direction = _load_direction(np, geometry_dir, layer, family)
    config = {
        "alpha_values": [float(value) for value in alphas],
        "batch_size": batch_size,
        "device": device,
        "family": family,
        "geometry_dir": str(geometry_dir),
        "layer": layer,
        "max_length": max_length,
        "max_new_tokens": max_new_tokens,
        "model_name": model_name,
        "prompt_file": None if prompt_file is None else str(prompt_file),
        "prompt_mode": prompt_mode,
        "resume_version": 1,
    }
    state = _ensure_checkpoint_state(
        output_dir=output_dir,
        name="sae_intervene",
        config=config,
        resume=resume,
    )
    state.setdefault("completed_batches", [])
    heartbeat.write_manifest(
        command="sae-intervene",
        geometry_dir=str(geometry_dir),
        layer=layer,
        model_name=model_name,
        prompt_count=len(prompts),
        prompt_mode=prompt_mode,
        resume=resume,
        target_family=family,
    )
    tokenizer, model = create_generation_components(model_name)
    runtime_device = _resolve_device(torch, device)
    model.to(runtime_device)
    model.eval()
    blocks = _find_transformer_blocks(model)
    if layer >= len(blocks):
        raise ValueError(f"Requested intervention layer {layer}, but model only exposes {len(blocks)} blocks.")
    rows: list[dict[str, Any]] = []
    total_batches = math.ceil(len(prompts) / batch_size)
    for batch_index, start in enumerate(range(0, len(prompts), batch_size), start=1):
        cached_rows = _load_intervention_batch_checkpoint(output_dir, batch_index) if resume else None
        if cached_rows is not None:
            rows.extend(cached_rows)
            completed_batches = set(int(value) for value in state.get("completed_batches", []))
            completed_batches.add(batch_index)
            state["completed_batches"] = sorted(completed_batches)
            _save_checkpoint_state(output_dir, "sae_intervene", state)
            heartbeat.update(
                phase="intervene",
                message=f"Loaded intervention batch {batch_index}/{total_batches} from checkpoint",
                batch_index=batch_index,
                total_batches=total_batches,
                processed_prompts=min(start + batch_size, len(prompts)),
                total_prompts=len(prompts),
            )
            continue
        batch_prompts = prompts[start : start + batch_size]
        rendered = [_render_prompt(tokenizer, row["prompt"]) for row in batch_prompts]
        tokenized = tokenizer(
            rendered,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = _move_batch_to_device(tokenized, runtime_device)
        last_positions = _non_padding_last_positions(encoded["attention_mask"])
        generation_kwargs: dict[str, Any] = {
            **encoded,
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
        }
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
        heartbeat.update(
            phase="intervene",
            message=f"Running intervention batch {batch_index}/{total_batches}",
            batch_index=batch_index,
            total_batches=total_batches,
            processed_prompts=min(start + batch_size, len(prompts)),
            total_prompts=len(prompts),
        )
        with torch.no_grad():
            baseline_generated = model.generate(**generation_kwargs)
        prompt_length = int(encoded["input_ids"].shape[1])
        baseline_completions = tokenizer.batch_decode(
            baseline_generated[:, prompt_length:].detach().cpu(),
            skip_special_tokens=True,
        )
        batch_rows: list[dict[str, Any]] = []
        for alpha in alphas:
            hook_state = {"applied": False}

            def patch_hook(_module: Any, _args: Any, output: Any) -> Any:
                hidden, remainder = _coerce_hidden_output(output)
                if hook_state["applied"] or getattr(hidden, "shape", None) is None:
                    return output
                patched = hidden.clone()
                patch_vector = torch.tensor(direction * float(alpha), device=patched.device, dtype=patched.dtype)
                for row_index, last_position in enumerate(last_positions):
                    patched[row_index, last_position, :] = patched[row_index, last_position, :] + patch_vector
                hook_state["applied"] = True
                if remainder:
                    return (patched, *remainder)
                return patched

            handle = blocks[layer].register_forward_hook(patch_hook)
            try:
                with torch.no_grad():
                    patched_generated = model.generate(**generation_kwargs)
            finally:
                handle.remove()
            patched_completions = tokenizer.batch_decode(
                patched_generated[:, prompt_length:].detach().cpu(),
                skip_special_tokens=True,
            )
            for prompt_row, baseline_raw, patched_raw in zip(
                batch_prompts,
                baseline_completions,
                patched_completions,
                strict=True,
            ):
                baseline_parsed = parse_format_completion(parse_format, baseline_raw)
                patched_parsed = parse_format_completion(parse_format, patched_raw)
                batch_rows.append(
                    {
                        "alpha": float(alpha),
                        "baseline_family": baseline_parsed.color_family,
                        "baseline_raw_completion": baseline_raw.strip(),
                        "changed": baseline_raw.strip() != patched_raw.strip(),
                        "layer": layer,
                        "matched_target_family": patched_parsed.color_family == family,
                        "patched_family": patched_parsed.color_family,
                        "patched_raw_completion": patched_raw.strip(),
                        "prompt": prompt_row["prompt"],
                        "prompt_id": prompt_row["prompt_id"],
                        "prompt_mode": prompt_mode,
                        "target_family": family,
                    }
                )
        rows.extend(batch_rows)
        _save_intervention_batch_checkpoint(output_dir, batch_index, batch_rows)
        completed_batches = set(int(value) for value in state.get("completed_batches", []))
        completed_batches.add(batch_index)
        state["completed_batches"] = sorted(completed_batches)
        _save_checkpoint_state(output_dir, "sae_intervene", state)
    _write_jsonl(output_dir / "intervention_rows.jsonl", rows)
    alpha_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    for alpha in alphas:
        alpha_group = [row for row in rows if float(row["alpha"]) == float(alpha)]
        matched_rate = float(
            sum(1 for row in alpha_group if row["matched_target_family"]) / len(alpha_group)
        ) if alpha_group else None
        changed_rate = float(sum(1 for row in alpha_group if row["changed"]) / len(alpha_group)) if alpha_group else None
        alpha_row = {
            "alpha": float(alpha),
            "changed_rate": changed_rate,
            "target_match_rate": matched_rate,
        }
        alpha_rows.append(alpha_row)
        if matched_rate is not None and (best_row is None or matched_rate > float(best_row["target_match_rate"])):
            best_row = alpha_row
    _write_jsonl(output_dir / "alpha_summary.jsonl", alpha_rows)
    summary = {
        "alphas": alphas,
        "best_alpha": None if best_row is None else float(best_row["alpha"]),
        "best_target_match_rate": None if best_row is None else float(best_row["target_match_rate"]),
        "geometry_dir": str(geometry_dir),
        "layer": layer,
        "model_name": model_name,
        "prompt_mode": prompt_mode,
        "target_family": family,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_intervention_report(output_dir / "report.md", summary=summary)
    _write_intervention_final_results(output_dir, summary=summary)
    heartbeat.update(phase="complete", message="SAE direction intervention complete", state="completed")
    return summary
