"""ESMC model loading and hidden-state helpers for ProtEnc."""

from __future__ import annotations

from typing import Any

import torch
from esm.models.esmc import ESMC

ESMC_HF_REPOS: dict[str, str] = {
    "esmc_300m": "biohub/ESMC-300M",
    "esmc_600m": "biohub/ESMC-600M",
    "esmc_6b": "biohub/ESMC-6B",
}

_HF_REPO_TO_MODEL = {repo.lower(): alias for alias, repo in ESMC_HF_REPOS.items()}


def resolve_esmc_repo_id(name: str) -> str:
    """Map a ProtEnc alias to the corresponding HuggingFace repo id."""
    if name in ESMC_HF_REPOS:
        return ESMC_HF_REPOS[name]
    lowered = name.lower()
    if lowered in _HF_REPO_TO_MODEL:
        return name
    for alias, repo in ESMC_HF_REPOS.items():
        if alias in lowered:
            return repo
    return name


def resolve_esmc_model_name(name: str) -> str:
    """Map user-facing names to ``ESMC.from_pretrained`` model ids."""
    if name in ESMC_HF_REPOS:
        return name
    lowered = name.lower()
    for alias in ESMC_HF_REPOS:
        if alias in lowered:
            return alias
    if lowered in _HF_REPO_TO_MODEL:
        return _HF_REPO_TO_MODEL[lowered]
    return name


def load_esmc(model_name: str, *, use_flash_attn: bool = True) -> tuple[ESMC, Any]:
    """Load an ESMC model and tokenizer via ``ESMC.from_pretrained``."""
    resolved_name = resolve_esmc_model_name(model_name)
    model = ESMC.from_pretrained(resolved_name)
    model.eval()
    return model, model.tokenizer


def esmc_num_layers(model: ESMC) -> int:
    return len(model.transformer.blocks)


def hf_config_num_layers(config: Any) -> int | None:
    """Read transformer layer count from a HuggingFace-style config when present."""
    for attr in ("num_hidden_layers", "n_layers", "num_layers"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    return None


def is_esmc_model(model: Any) -> bool:
    return isinstance(model, ESMC)


def is_esmc_hf_model(model: Any) -> bool:
    """Backward-compatible alias used by ProteinEncoder."""
    return is_esmc_model(model)


def _sequence_tokens(model: ESMC, inputs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(inputs, dict):
        return inputs["input_ids"]
    return inputs


def _attention_mask(
    model: ESMC, sequence_tokens: torch.Tensor, inputs: torch.Tensor | dict[str, torch.Tensor]
) -> torch.Tensor:
    if isinstance(inputs, dict) and "attention_mask" in inputs:
        return inputs["attention_mask"].bool()
    pad_idx = model.tokenizer.pad_token_id
    return sequence_tokens != pad_idx


def esmc_hidden_states(
    model: ESMC, inputs: torch.Tensor | dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Return stacked ESMC hidden states with shape ``[n_layers + 1, B, L, D]``.

    Index ``0`` is the embedding layer; indices ``1..n_layers`` are transformer blocks.
    This matches ProtEnc's 1-indexed ``repr_layer`` convention used for ESM2 models.
    """
    sequence_tokens = _sequence_tokens(model, inputs)
    attention_mask = _attention_mask(model, sequence_tokens, inputs)

    embedded = model.embed(sequence_tokens)
    output = model(sequence_tokens=sequence_tokens, sequence_id=attention_mask)
    assert output.hidden_states is not None

    # Prepend the embedding layer so repr_layer=L selects hidden_states[L].
    return torch.cat([embedded.unsqueeze(0), output.hidden_states], dim=0)


def esmc_select_hidden_layer(hidden_states: torch.Tensor, repr_layer: int) -> torch.Tensor:
    """Select a 1-indexed representation layer from stacked ESMC hidden states."""
    return hidden_states[repr_layer]


def esmc_hidden_at_layer(
    model: ESMC,
    inputs: torch.Tensor | dict[str, torch.Tensor],
    repr_layer: int,
) -> torch.Tensor:
    hidden_states = esmc_hidden_states(model, inputs)
    return esmc_select_hidden_layer(hidden_states, repr_layer)


def tokenize_esmc_sequences(model: ESMC, sequences: list[str]) -> torch.Tensor:
    """Tokenize protein sequences for ESMC inference."""
    return model._tokenize(sequences)
