from __future__ import annotations

from typing import Any

SUPPORTED_GENERATION_MODELS = {
    "meta-llama/Llama-3.1-8B",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-v0.3",
}


def _require_ml_stack() -> tuple[Any, Any]:
    try:
        import torch  # type: ignore[import-not-found]
        import transformers  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "The ML stack is not installed. Run `pip install -e .` in the repo environment."
        ) from exc
    return torch, transformers


def _validate_model_name(model_name: str) -> None:
    if model_name not in SUPPORTED_GENERATION_MODELS:
        supported = ", ".join(sorted(SUPPORTED_GENERATION_MODELS))
        raise ValueError(f"Unsupported model {model_name!r}. Expected one of: {supported}")


def get_tokenizer(model_name: str) -> Any:
    _torch, transformers = _require_ml_stack()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def create_generation_components(model_name: str) -> tuple[Any, Any]:
    _validate_model_name(model_name)
    _torch, transformers = _require_ml_stack()
    tokenizer = get_tokenizer(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
        torch_dtype="auto",
    )
    return tokenizer, model
