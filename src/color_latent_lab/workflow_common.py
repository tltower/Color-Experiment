from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .word_lists import default_words, find_system_word_list, preset_words, read_word_file


def _require_stack() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import numpy as np  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        from sklearn.decomposition import PCA  # type: ignore[import-not-found]
        from sklearn.linear_model import LogisticRegression  # type: ignore[import-not-found]
        from sklearn.model_selection import KFold  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "The research stack is not installed. Use `pip install -e .` in the repo environment."
        ) from exc
    return np, torch, PCA, LogisticRegression, KFold


def _select_layers(hidden_states: tuple[Any, ...], requested_layers: tuple[int, ...] | None) -> tuple[int, ...]:
    if requested_layers is None:
        return tuple(range(len(hidden_states)))
    valid_layers: list[int] = []
    upper_bound = len(hidden_states) - 1
    for layer in requested_layers:
        if layer < 0 or layer > upper_bound:
            raise ValueError(f"Requested layer {layer} outside valid range 0..{upper_bound}")
        valid_layers.append(layer)
    return tuple(dict.fromkeys(valid_layers))


def _read_words(
    word_list_path: Path | None,
    limit: int | None,
    *,
    word_preset: str = "default",
) -> tuple[list[str], str]:
    if word_list_path is not None:
        source = str(word_list_path)
        words = read_word_file(word_list_path, limit=limit)
    elif word_preset == "color_words":
        source = "color_words"
        words = preset_words("color_words", limit=limit)
    else:
        default_seed = default_words()
        if limit is None or limit <= len(default_seed):
            source = word_preset
            words = preset_words(word_preset, limit=limit)
        else:
            system_dictionary = find_system_word_list()
            if system_dictionary is None:
                raise RuntimeError(
                    f"Requested {limit} words, but only {len(default_seed)} built-in words are available. "
                    "Install a system dictionary or pass `--word-list-path`."
                )
            source = str(system_dictionary)
            words = read_word_file(system_dictionary, limit=limit)
    if not words:
        raise ValueError("Word list resolved to zero words.")
    return words, source


def _mean_or_none(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


__all__ = [
    "_mean_or_none",
    "_read_words",
    "_require_stack",
    "_select_layers",
]
