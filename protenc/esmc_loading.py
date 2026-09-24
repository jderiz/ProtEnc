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


def _ensure_esmc_tokenizer_hub_alias() -> None:
    """Hub configs name ``EsmcTokenizer``; Biohub transformers exports ``ESMCTokenizer``."""
    try:
        import transformers
    except ImportError:
        return
    existing = getattr(transformers, "EsmcTokenizer", None)
    if existing is not None and existing.__name__ == "EsmcTokenizer":
        return
    base = getattr(transformers, "ESMCTokenizer", None)
    if base is None:
        try:
            from transformers.models.esmc.tokenization_esmc import ESMCTokenizer as base
        except ImportError:
            return

    class EsmcTokenizer(base):  # type: ignore[valid-type,misc]
        pass

    setattr(transformers, "EsmcTokenizer", EsmcTokenizer)


_ESMC_LEGACY_PTH: dict[str, dict[str, Any]] = {
    "esmc_300m": {
        "repo": "biohub/esmc-300m-2024-12",
        "d_model": 960,
        "n_heads": 15,
        "n_layers": 30,
    },
    "esmc_600m": {
        "repo": "biohub/esmc-600m-2024-12",
        "d_model": 1152,
        "n_heads": 18,
        "n_layers": 36,
    },
    "esmc_6b": {
        "repo": "biohub/esmc-6b-2024-12",
        "d_model": 2560,
        "n_heads": 40,
        "n_layers": 80,
    },
}


def _load_esmc_legacy_nested_pth(
    alias: str,
    *,
    use_flash_attn: bool = True,
    device: torch.device | None = None,
) -> ESMC:
    """Load ESMC from ``biohub/esmc-*-2024-12`` nested ``data/weights/*.pth``.

    esm 3.3 + huggingface_hub>=0.36 fails two ways on the official builders:
    1. ``load_torch_model(snapshot_dir)`` rejects the directory (weights are nested).
    2. Even with a ``.pth`` path, ``init_empty_weights`` + ``load_state_dict`` without
       ``assign=True`` leaves meta tensors and ``.to(device)`` raises.

    Materialize a real module, load the state dict, then move to device (bf16 on GPU).
    """
    from pathlib import Path

    from esm.tokenization import get_esmc_model_tokenizers
    from huggingface_hub import snapshot_download

    if alias not in _ESMC_LEGACY_PTH:
        raise KeyError(f"No legacy ESMC pth spec for {alias!r}")
    spec = _ESMC_LEGACY_PTH[alias]
    snap = Path(snapshot_download(repo_id=spec["repo"]))
    pths = sorted((snap / "data" / "weights").glob("*.pth"))
    if len(pths) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .pth under {snap / 'data' / 'weights'}, found {pths!r}"
        )

    model = ESMC(
        d_model=int(spec["d_model"]),
        n_heads=int(spec["n_heads"]),
        n_layers=int(spec["n_layers"]),
        tokenizer=get_esmc_model_tokenizers(),
        use_flash_attn=use_flash_attn,
    ).eval()
    state = torch.load(pths[0], map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state and "embed.weight" not in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(
            f"ESMC legacy pth load missing keys for {alias}: {missing[:8]}..."
        )
    if unexpected:
        # Non-fatal for auxiliary buffers; still surface for debugging.
        pass

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if device.type != "cpu":
        model = model.to(torch.bfloat16)
    return model


def load_esmc(model_name: str, *, use_flash_attn: bool = True) -> tuple[ESMC, Any]:
    """Load an ESMC model and tokenizer via ``ESMC.from_pretrained``."""
    _ensure_esmc_tokenizer_hub_alias()
    resolved_name = resolve_esmc_model_name(model_name)
    try:
        try:
            model = ESMC.from_pretrained(resolved_name, use_flash_attn=use_flash_attn)
        except TypeError:
            # Older ESMC.from_pretrained signatures omit use_flash_attn.
            model = ESMC.from_pretrained(resolved_name)
    except (ValueError, NotImplementedError):
        if resolved_name not in _ESMC_LEGACY_PTH:
            raise
        model = _load_esmc_legacy_nested_pth(
            resolved_name, use_flash_attn=use_flash_attn
        )
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


def _sequence_tokens(
    model: ESMC, inputs: torch.Tensor | dict[str, torch.Tensor]
) -> torch.Tensor:
    if isinstance(inputs, dict):
        return inputs["input_ids"]
    return inputs


def _attention_mask(
    model: ESMC,
    sequence_tokens: torch.Tensor,
    inputs: torch.Tensor | dict[str, torch.Tensor],
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


def esmc_select_hidden_layer(
    hidden_states: torch.Tensor, repr_layer: int
) -> torch.Tensor:
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
