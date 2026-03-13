from __future__ import annotations

from typing import Any


def _render_prompt(tokenizer: Any, prompt: str) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            return str(
                apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except Exception:
            return prompt
    return prompt


def _resolve_device(torch: Any, requested_device: str) -> Any:
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _non_padding_last_positions(attention_mask: Any) -> list[int]:
    positions: list[int] = []
    for row in attention_mask.detach().cpu().tolist():
        indices = [index for index, value in enumerate(row) if int(value) == 1]
        if not indices:
            raise RuntimeError("Encountered an empty prompt after tokenization.")
        positions.append(indices[-1])
    return positions


def _find_transformer_blocks(model: Any) -> Any:
    for candidate in (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
        getattr(getattr(model, "gpt_neox", None), "layers", None),
    ):
        if candidate is not None:
            return candidate
    raise RuntimeError("Could not find transformer blocks for residual patching on this model.")


def _coerce_hidden_output(output: Any) -> tuple[Any, tuple[Any, ...]]:
    if isinstance(output, tuple):
        return output[0], tuple(output[1:])
    return output, ()


__all__ = [
    "_coerce_hidden_output",
    "_find_transformer_blocks",
    "_move_batch_to_device",
    "_non_padding_last_positions",
    "_render_prompt",
    "_resolve_device",
]
