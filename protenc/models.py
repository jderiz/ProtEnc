import contextlib
import os
import re
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional

import colorlog as logging
import numpy as np
import torch
import torch.nn as nn
from esm.models.esm3 import ESM3
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, ProteinComplex
from esm.utils.structure.protein_chain import ProteinChain
from sequence_models.constants import GAP
from sequence_models.pretrained import load_model_and_alphabet
from transformers import (
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    EsmModel,
    T5EncoderModel,
    T5Tokenizer,
)

from protenc.esmc_loading import (
    ESMC_HF_REPOS,
    esmc_hidden_states,
    esmc_num_layers,
    esmc_select_hidden_layer,
    hf_config_num_layers,
    load_esmc,
    tokenize_esmc_sequences,
)
from protenc.saprot import chain_states as saprot_chain_states
from protenc.saprot import combine as saprot_combine
from protenc.saprot import sequence_only as saprot_sequence_only

try:
    from .mpnn import (
        ProteinMPNN,
        cat_neighbors_nodes,
        gather_nodes,
        parse_PDB,
        tied_featurize,
    )
except ImportError:
    from mpnn import (  # pyright: ignore[reportMissingImports]
        ProteinMPNN,
        cat_neighbors_nodes,
        gather_nodes,
        parse_PDB,
        tied_featurize,
    )

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

_ESMC_NUM_LAYERS = {
    "esmc_300m": 30,
    "esmc_600m": 36,
    "esmc_6b": 80,
}
for _alias, _repo in ESMC_HF_REPOS.items():
    _ESMC_NUM_LAYERS[_repo.lower()] = _ESMC_NUM_LAYERS[_alias]


def resolve_repr_layer(repr_layer: int | None, num_layers: int) -> int:
    return num_layers if repr_layer is None else repr_layer


def validate_repr_layer(repr_layer: int, num_layers: int, *, model_name: str) -> None:
    if not 1 <= repr_layer <= num_layers:
        warnings.warn(
            f"repr_layer={repr_layer} is invalid for '{model_name}': "
            f"valid range is 1..{num_layers}.",
            stacklevel=2,
        )
        raise ValueError(
            f"repr_layer={repr_layer} is invalid for '{model_name}': "
            f"valid range is 1..{num_layers}."
        )


def parse_repr_layers_config(
    repr_layer: int | None,
    repr_layers: list[int] | None,
    num_layers: int,
    *,
    model_name: str,
) -> list[int]:
    """
    Normalize ``repr_layer`` / ``repr_layers`` config into a sorted unique layer list.

    When ``repr_layers`` is provided it takes precedence and ``repr_layer`` is ignored.
    Otherwise a single layer is resolved from ``repr_layer`` (or the model's final layer).
    Every returned layer is validated against ``num_layers``.
    """
    if repr_layers:
        layers = sorted({int(layer) for layer in repr_layers})
    else:
        layers = [resolve_repr_layer(repr_layer, num_layers)]

    for layer in layers:
        validate_repr_layer(layer, num_layers, model_name=model_name)

    return layers


def layer_output_path(path, layer: int):
    """Insert a ``_layer{N}`` suffix before the file extension of ``path``."""
    from pathlib import Path

    p = Path(path)
    new_name = f"{p.stem}_layer{layer}{p.suffix}"
    return (
        type(path)(p.with_name(new_name))
        if isinstance(path, str)
        else p.with_name(new_name)
    )


def _esm2_num_layers_from_name(name: str) -> int:
    match = re.search(r"esm2_t(\d+)", name, re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Cannot determine ESM2 layer count from name '{name}'. "
            "Expected pattern esm2_t<N>_..."
        )
    return int(match.group(1))


def _esmc_num_layers_from_name(name: str) -> int:
    lowered = name.lower()
    for variant, num_layers in _ESMC_NUM_LAYERS.items():
        if variant in lowered:
            return num_layers
    warnings.warn(f"Unknown ESMC variant in name '{name}'.")
    raise ValueError(
        f"Unknown ESMC variant in name '{name}'. "
        f"Expected one of {sorted(_ESMC_NUM_LAYERS)}."
    )


def _hf_hidden_at_layer(hidden_states: tuple, repr_layer: int) -> torch.Tensor:
    return hidden_states[repr_layer]


def _esm3_hidden_at_layer(hidden_states: tuple, repr_layer: int) -> torch.Tensor:
    return hidden_states[repr_layer - 1]


def _hf_core(backbone: nn.Module, inputs: dict, layers: tuple[int, ...]):
    hidden_states = backbone(**inputs, output_hidden_states=True).hidden_states
    return torch.stack(
        [_hf_hidden_at_layer(hidden_states, layer) for layer in layers], dim=1
    )


def _esmc_core(
    backbone: nn.Module,
    inputs: dict,
    layers: tuple[int, ...],
    use_norm_layer: bool = True,
):
    hidden_states = esmc_hidden_states(backbone, inputs)  # [n_layers + 1, B, L, D]
    if use_norm_layer:
        # Index 0 is the embedding layer, so the last block is index n_layers.
        num_layers = hidden_states.shape[0] - 1
        norm = backbone.transformer.norm
        selected = []
        for layer in layers:
            embeddings = esmc_select_hidden_layer(hidden_states, layer)
            if layer == num_layers:
                embeddings = norm(embeddings.to(next(norm.parameters()).dtype))
            selected.append(embeddings.float())
        return torch.stack(selected, dim=1)
    return torch.stack(
        [esmc_select_hidden_layer(hidden_states, layer) for layer in layers], dim=1
    )


def _carp_core(backbone: nn.Module, inputs: dict, layers: tuple[int, ...]):
    results = backbone(inputs["tokens"], repr_layers=list(layers), logits=False)
    return torch.stack([results["representations"][layer] for layer in layers], dim=1)


def _assert_esmc_num_layers(model: ESMC, expected: int, *, model_name: str) -> None:
    actual = esmc_num_layers(model)
    if actual != expected:
        warnings.warn(
            f"Layer count mismatch for '{model_name}': "
            f"expected {expected}, model reports {actual}."
        )
        raise ValueError(
            f"Layer count mismatch for '{model_name}': "
            f"expected {expected}, model reports {actual}."
        )


def _assert_hf_num_hidden_layers(
    model: nn.Module, expected: int, *, model_name: str
) -> None:
    actual = hf_config_num_layers(model.config)
    if actual is None:
        raise ValueError(f"Could not read layer count from config for '{model_name}'.")
    if actual != expected:
        warnings.warn(
            f"Layer count mismatch for '{model_name}': "
            f"expected {expected}, config reports {actual}."
        )
        raise ValueError(
            f"Layer count mismatch for '{model_name}': "
            f"expected {expected}, config reports {actual}."
        )


class EmbeddingType(Enum):
    PER_RESIDUE = "per_residue"
    PER_PROTEIN = "per_protein"


class StructureInput(Enum):
    """Whether a model takes backbone structures alongside the sequences."""

    NONE = "none"
    OPTIONAL = "optional"
    REQUIRED = "required"


class _HiddenCore(nn.Module):
    """
    DataParallel-safe compute core of an embedder.

    ``forward(inputs, layers)`` returns a batch-first tensor ``[B, len(layers), L, D]``
    so ``nn.DataParallel`` can scatter ``inputs`` and gather the output along dim 0.
    ``fn`` must only use the ``backbone`` it is given (each replica passes its own copy),
    and ``static`` must hold plain values shared across replicas.
    """

    def __init__(self, backbone: nn.Module, fn: Callable[..., torch.Tensor], **static):
        super().__init__()
        self.backbone = backbone
        self.fn = fn
        self.static = static

    @torch.no_grad()
    def forward(self, inputs: dict, layers: tuple[int, ...]) -> torch.Tensor:
        return self.fn(self.backbone, inputs, layers, **self.static)


class BaseProteinEmbeddingModel(nn.Module):
    embedding_type: EmbeddingType
    # What the model needs besides sequences; advertised by ``ProteinEncoder.requirements``.
    structure_input: StructureInput = StructureInput.NONE
    # Default: no explicit chain break token. Specific models may override.
    chain_break_token: str = ""
    # Layers to extract in the multi-layer path; set by get_model when configured.
    repr_layers: list[int] | None = None
    # Implemented by subclasses that support multi-layer extraction (type-only
    # declaration; no runtime attribute is created here).
    forward_multi: Callable[..., Any]

    @property
    def structure_aware(self) -> bool:
        return self.structure_input is not StructureInput.NONE

    def prepare_sequences(self, sequences, structures=None) -> Any:
        return NotImplementedError

    def forward(self, input):
        raise NotImplementedError

    def _set_core(self, fn: Callable[..., torch.Tensor], **static) -> None:
        """Build the compute core around ``self.model``.

        The core is stored outside ``_modules`` so the backbone is not registered twice
        (no duplicate ``state_dict`` keys); it shares the backbone with ``self.model``,
        so ``.to()``/``.eval()`` on the embedder still apply to it.
        """
        object.__setattr__(self, "_core", _HiddenCore(self.model, fn, **static))

    def enable_data_parallel(self, device_ids: list[int] | None = None) -> None:
        """Replicate the compute core across GPUs with ``nn.DataParallel``."""
        if isinstance(self._core, nn.DataParallel):
            return
        object.__setattr__(
            self, "_core", nn.DataParallel(self._core, device_ids=device_ids)
        )

    @property
    def is_data_parallel(self) -> bool:
        return isinstance(getattr(self, "_core", None), nn.DataParallel)

    def _run_core(self, input, layers) -> torch.Tensor:
        """Run the compute core on a prepared batch; returns ``[B, len(layers), L, D]``."""
        if isinstance(input, torch.Tensor):
            input = {"input_ids": input}
        # Plain dict (not BatchEncoding/UserDict) so DataParallel scatters its tensors.
        device = next(self.model.parameters()).device
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in dict(input).items()
        }
        if self.is_data_parallel:
            _drop_self_bound_forwards(self.model)
        return self._core(inputs, tuple(layers))


def _drop_self_bound_forwards(module: nn.Module) -> None:
    """Remove instance-level ``forward`` attributes that just re-bind the class forward.

    transformers' ``check_model_inputs`` (output capturing for ``output_hidden_states``)
    "restores" patched layers with ``module.forward = <bound method of module>``, leaving
    an instance attribute behind after the first call. ``nn.DataParallel.replicate``
    shallow-copies ``__dict__``, so every replica would then call the original module's
    forward with its cuda:0 weights. Deleting the attribute is a no-op for normal calls.
    """
    for m in module.modules():
        fwd = m.__dict__.get("forward")
        if (
            fwd is not None
            and getattr(fwd, "__self__", None) is m
            and getattr(fwd, "__func__", None) is type(m).forward
        ):
            del m.__dict__["forward"]


def load_huggingface_language_model(
    model_cls, tokenizer_cls, model_name, load_weights=True
):
    if load_weights:
        model = model_cls.from_pretrained(model_name)
        tokenizer = tokenizer_cls.from_pretrained(model_name)

        return model, tokenizer
    else:
        config = model_cls.config_class.from_pretrained(model_name)
        model = model_cls(config)
        tokenizer = tokenizer_cls.from_pretrained(model_name)

        return model, tokenizer


class BaseProtTransEmbeddingModel(BaseProteinEmbeddingModel):
    embedding_kind = EmbeddingType.PER_RESIDUE
    available_models = None

    def __init__(
        self,
        model,
        tokenizer,
        repr_layer: int | None = None,
        model_name: str = "",
    ):
        super().__init__()
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        num_layers = model.config.num_hidden_layers
        self.repr_layer = resolve_repr_layer(repr_layer, num_layers)
        validate_repr_layer(
            self.repr_layer, num_layers, model_name=model_name or type(self).__name__
        )
        self._set_core(_hf_core)

    def _validate_model_name(self, model_name):
        assert self.available_models is None or model_name in self.available_models, (
            f"Unknown model name '{model_name}'. Available options are {self.available_models}"
        )

    def prepare_sequences(self, sequences, structures=None):
        sequences = [" ".join(s.replace(" ", "")) for s in sequences]

        return self.tokenizer.batch_encode_plus(
            sequences, return_tensors="pt", add_special_tokens=True, padding=True
        )

    def _post_process_embedding(self, embed, seq_len):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        hidden = self._run_core(input, (self.repr_layer,))
        seq_lens = (input["attention_mask"] == 1).sum(-1)

        for i, seq_len in enumerate(seq_lens):
            yield self._post_process_embedding(hidden[i, 0].cpu(), seq_len)

    @torch.no_grad()
    def forward_multi(self, input, repr_layers: list[int]):
        hidden = self._run_core(input, repr_layers)
        seq_lens = (input["attention_mask"] == 1).sum(-1)

        for i, seq_len in enumerate(seq_lens):
            for j, layer in enumerate(repr_layers):
                yield (
                    i,
                    layer,
                    self._post_process_embedding(hidden[i, j].cpu(), seq_len),
                )


class ProtBERTEmbeddingModel(BaseProtTransEmbeddingModel):
    available_models = ["prot_bert", "prot_bert_bfd"]

    def __init__(
        self, model_name: str, load_weights: bool = True, repr_layer: int | None = None
    ):
        self._validate_model_name(model_name)

        mode_name = f"Rostlab/{model_name}"
        model, tokenizer = load_huggingface_language_model(
            BertModel, BertTokenizer, mode_name, load_weights=load_weights
        )

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            repr_layer=repr_layer,
            model_name=model_name,
        )

    def _post_process_embedding(self, embed, seq_len):
        return embed[1 : seq_len - 1]


class ProtT5EmbeddingModel(BaseProtTransEmbeddingModel):
    available_models = [
        "prot_t5_xl_uniref50",
        "prot_t5_xl_bfd",
        "prot_t5_xxl_uniref50",
        "prot_t5_xxl_bfd",
    ]

    def __init__(
        self, model_name: str, load_weights: bool = True, repr_layer: int | None = None
    ):
        self._validate_model_name(model_name)

        mode_name = f"Rostlab/{model_name}"
        model, tokenizer = load_huggingface_language_model(
            T5EncoderModel, T5Tokenizer, mode_name, load_weights=load_weights
        )

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            repr_layer=repr_layer,
            model_name=model_name,
        )

    def _post_process_embedding(self, embed, seq_len):
        return embed[: seq_len - 1]


class ESMEmbeddingModel(BaseProteinEmbeddingModel):
    embedding_kind = EmbeddingType.PER_RESIDUE
    # ESM2-style models in HAIPR use "<eos>" as chain break token.
    chain_break_token: str = "<eos>"

    def __init__(self, model_name: str, repr_layer: int | None = None):
        super().__init__()

        num_layers = _esm2_num_layers_from_name(model_name)
        self.repr_layer = resolve_repr_layer(repr_layer, num_layers)
        validate_repr_layer(self.repr_layer, num_layers, model_name=model_name)

        self.model = EsmModel.from_pretrained("facebook/" + model_name)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/" + model_name)

        self.model.eval()
        _assert_hf_num_hidden_layers(self.model, num_layers, model_name=model_name)
        self._set_core(_hf_core)

    def clean(self, seq):
        if not re.match(r"^[ACDEFGHIKLMNPQRSTVWYX]+$", seq):
            print(f"Invalid sequence: {seq}")

            seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWYX]", "X", seq)
            print(f"Converted sequence: {seq}")
        return seq

    def prepare_sequences(self, sequences, structures=None):
        batch_tokens = self.tokenizer(
            sequences, return_tensors="pt", add_special_tokens=True, padding=True
        )

        return batch_tokens

    @torch.no_grad()
    def forward(self, input):
        logger.debug(f"Input: {input}")
        hidden = self._run_core(input, (self.repr_layer,))
        seq_lengths = input["attention_mask"].sum(1)

        for i, seq_len in enumerate(seq_lengths):
            yield hidden[i, 0, 1 : seq_len - 1]

    @torch.no_grad()
    def forward_multi(self, input, repr_layers: list[int]):
        logger.debug(f"Input: {input}")
        hidden = self._run_core(input, repr_layers)
        seq_lengths = input["attention_mask"].sum(1)

        for i, seq_len in enumerate(seq_lengths):
            for j, layer in enumerate(repr_layers):
                yield i, layer, hidden[i, j, 1 : seq_len - 1]


class SaProtEmbeddingModel(BaseProteinEmbeddingModel):
    """SaProt: an ESM2-architecture model over residue + Foldseek 3Di tokens.

    Without structures every 3Di slot is ``#`` (masked); with a structure file the
    states are computed by ``mini3di``. See ``protenc.saprot``.
    """

    embedding_kind = EmbeddingType.PER_RESIDUE
    structure_input = StructureInput.OPTIONAL
    # Same ESM2 vocabulary and special tokens as ESMEmbeddingModel.
    chain_break_token: str = "<eos>"

    def __init__(
        self,
        model_name: str,
        num_layers: int,
        repr_layer: int | None = None,
        plddt_mask: bool = False,
    ):
        super().__init__()
        self.repr_layer = resolve_repr_layer(repr_layer, num_layers)
        validate_repr_layer(self.repr_layer, num_layers, model_name=model_name)
        self.plddt_mask = plddt_mask

        self.model = EsmModel.from_pretrained(model_name, add_pooling_layer=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model.eval()
        _assert_hf_num_hidden_layers(self.model, num_layers, model_name=model_name)
        self._set_core(_hf_core)

    def prepare_sequences(
        self, sequences, structures=None, chain_list=None, structure_id=None
    ):
        """
        Tokenize ``sequences`` as SaProt residue + 3Di tokens.

        Args:
            sequences: Amino-acid sequences; chains separated by ``|`` or ``<eos>``.
            structures: PDB/mmCIF path (string or single-element list) to take the
                3Di states from. ``None`` masks every state.
            chain_list: Chains of the structure, in the order of the sequence's
                chains. ``None`` takes every chain in file order.
            structure_id: Unused; accepted so callers can pass it to any embedder.

        With a structure, each chain must have one residue per CA-bearing residue
        of the structure's chain.
        """
        brk = self.chain_break_token
        sequences = [s.replace("|", brk) for s in sequences]
        path = structures if isinstance(structures, str) else None
        if path is None and structures:
            path = structures[0]

        if path is None:
            tokens = [
                brk.join(saprot_sequence_only(c) for c in s.split(brk))
                for s in sequences
            ]
        else:
            states = saprot_chain_states(path, chain_list, self.plddt_mask)
            tokens = []
            for s in sequences:
                chains = s.split(brk)
                if len(chains) != len(states) or any(
                    len(c) != len(st) for c, st in zip(chains, states)
                ):
                    raise ValueError(
                        f"Sequence chains {[len(c) for c in chains]} do not match the "
                        f"structure's chains {[len(st) for st in states]} in {path}"
                    )
                tokens.append(
                    brk.join(saprot_combine(c, st) for c, st in zip(chains, states))
                )
        return self.tokenizer(
            tokens, return_tensors="pt", add_special_tokens=True, padding=True
        )

    @torch.no_grad()
    def forward(self, input):
        hidden = self._run_core(input, (self.repr_layer,))
        seq_lengths = input["attention_mask"].sum(1)

        for i, seq_len in enumerate(seq_lengths):
            yield hidden[i, 0, 1 : seq_len - 1]

    @torch.no_grad()
    def forward_multi(self, input, repr_layers: list[int]):
        hidden = self._run_core(input, repr_layers)
        seq_lengths = input["attention_mask"].sum(1)

        for i, seq_len in enumerate(seq_lengths):
            for j, layer in enumerate(repr_layers):
                yield i, layer, hidden[i, j, 1 : seq_len - 1]


def _create_filtered_protein_complex(
    pdb_path: str, chain_list: List[str], id: Optional[str] = None
) -> ProteinComplex:
    """
    Load a ProteinComplex from a PDB file with only the specified chains.
    Mirrors create_filtered_protein_complex from haipr/models/esm3.py.
    """
    chains = []
    for chain_id in chain_list:
        try:
            chain = ProteinChain.from_pdb(pdb_path, chain_id=chain_id, id=id)
            chains.append(chain)
            logger.debug(
                f"Successfully loaded chain {chain_id} with {len(chain.sequence)} residues"
            )
        except Exception as e:
            logger.warning(f"Failed to load chain {chain_id} from PDB: {e}")
            continue

    if not chains:
        raise ValueError(
            f"No valid chains found for chain_list {chain_list} in PDB file {pdb_path}"
        )

    return ProteinComplex.from_chains(chains)


class ESM3EmbeddingModel(BaseProteinEmbeddingModel):
    """ESM3 embedder; batches must contain sequences of the same length (no padding)."""

    embedding_kind = EmbeddingType.PER_RESIDUE
    structure_input = StructureInput.OPTIONAL
    requires_same_length_batch = True
    # ProtEnc ESM3 expects "|" as the multi-chain separator.
    chain_break_token: str = "|"

    def __init__(
        self,
        model_name: str,
        use_norm_layer: bool = True,
        repr_layer: int | None = None,
    ):
        super().__init__()
        self.model: ESM3 = ESM3.from_pretrained(model_name)
        self.model.eval()
        self.use_norm_layer = use_norm_layer
        self.model_name = model_name
        num_layers = len(self.model.transformer.blocks)
        self.repr_layer = resolve_repr_layer(repr_layer, num_layers)
        validate_repr_layer(self.repr_layer, num_layers, model_name=model_name)
        self._set_core(_esm3_core, use_norm_layer=use_norm_layer)

    def prepare_sequences(
        self,
        sequences: List[str],
        structures=None,
        chain_list: Optional[List[str]] = None,
        structure_id: Optional[str] = None,
    ):
        """
        Prepare sequences and structures for ESM3 embedding.

        Uses default ESM3 model.encode() for tokenization in both sequence-only and
        sequence+structure paths. Collation yields a batched dict with sequence_tokens
        and optional structure_tokens for forward. All sequences in a batch must have
        the same length (stacked, no padding).

        Args:
            sequences: List of protein sequences (may use "|" for multi-chain).
            structures: Optional path to structure file for structure-aware encoding
                (string or single-element list).
            chain_list: Optional list of chain IDs to load (None = full complex).
            structure_id: Optional structure id passed to ProteinChain.from_pdb when
                loading by chain_list (mirrors id in haipr create_filtered_protein_complex).

        Returns:
            Dict with "sequence_tokens", optional "structure_tokens", for forward.
        """
        logger.info("Preparing sequences for ESM3")
        structure_path = (
            structures
            if isinstance(structures, str)
            else (structures[0] if structures else None)
        )
        use_structure = structure_path is not None
        model = self.model

        if structure_path is None:
            # Sequence-only: ESMProtein(sequence=...) then default model.encode()
            logger.info("Sequence-only path (use_structure=False)")
            proteins = [ESMProtein(sequence=seq) for seq in sequences]
            protein_tensors = [model.encode(p) for p in proteins]
        else:
            try:
                if chain_list is not None:
                    pc = _create_filtered_protein_complex(
                        structure_path, chain_list, id=structure_id
                    )
                else:
                    pc = ProteinComplex.from_pdb(structure_path)

                protein_with_structure = ESMProtein.from_protein_complex(pc)
                shared_coordinates = protein_with_structure.coordinates

                if chain_list is not None:
                    # Fast path: HAIPRData guarantees sequences are already in chain_list
                    # order (split on "|"). ProteinComplex.from_chains inserts one
                    # chain-break row between chains, so keep "|" to match
                    # shared_coordinates and the sequence-only token layout.
                    flat_sequences = list(sequences)
                else:
                    # Slow path: align sequences to PDB chain order via biotite.
                    flat_sequences = []
                    try:
                        from biotite.structure.io.pdb import PDBFile as _PDBFile

                        pdb_file = _PDBFile.read(structure_path)
                        structure = pdb_file.get_structure()
                        models = (
                            structure[0]
                            if hasattr(structure, "stack_depth")
                            and structure.stack_depth() > 0
                            else structure
                        )
                        all_chain_ids = []
                        for cid in models.chain_id:
                            if cid not in all_chain_ids:
                                all_chain_ids.append(cid)
                        loaded_chain_ids = [c.chain_id for c in pc.chain_iter()]
                        logger.debug(
                            f"PDB chains: {all_chain_ids}, ProteinComplex loaded chains: {loaded_chain_ids}"
                        )
                        for seq in sequences:
                            parts = seq.split("|")
                            if len(parts) == len(all_chain_ids):
                                chain_to_seq = dict(zip(all_chain_ids, parts))
                                aligned_parts = [
                                    chain_to_seq.get(c.chain_id, "") or str(c.sequence)
                                    for c in pc.chain_iter()
                                ]
                                flat_sequences.append("|".join(aligned_parts))
                            elif len(parts) == len(loaded_chain_ids):
                                flat_sequences.append(seq)
                            else:
                                logger.warning(
                                    f"Sequence parts ({len(parts)}) don't match PDB chains "
                                    f"({len(all_chain_ids)}) or loaded chains ({len(loaded_chain_ids)}). "
                                    "Falling back to stripping chain-break tokens."
                                )
                                flat_sequences.append(seq.replace("|", ""))
                    except Exception as e:
                        logger.warning(
                            f"Failed to align sequences using biotite: {e}; "
                            "falling back to stripping chain-break tokens."
                        )
                        flat_sequences = [seq.replace("|", "") for seq in sequences]

                if shared_coordinates is not None:
                    n_coords = shared_coordinates.shape[0]
                    bad = [s for s in flat_sequences if len(s) != n_coords]
                    if bad:
                        raise ValueError(
                            f"{len(bad)} sequence(s) do not match the structure length "
                            f"{n_coords} (first has length {len(bad[0])}); chains "
                            f"{[c.chain_id for c in pc.chain_iter()]}"
                        )

                proteins = [
                    ESMProtein(sequence=flat_seq, coordinates=shared_coordinates)
                    for flat_seq in flat_sequences
                ]
                first = proteins[0]
                if first.coordinates is None:
                    protein_tensors = [model.encode(p) for p in proteins]
                    use_structure = False
                else:
                    protein_tensors = [model.encode(p) for p in proteins]
            except Exception as e:
                raise ValueError(f"Failed to prepare sequences for ESM3: {e}")

        # Collate like haipr prepare_training_features: sequence_tokens_list, structure_tokens_list
        sequence_tokens_list = []
        structure_tokens_list = []
        for p in protein_tensors:
            seq_tok = (
                p.sequence.cpu()
                if isinstance(p.sequence, torch.Tensor) and p.sequence.is_cuda
                else p.sequence
            )
            struct_tok = (
                p.structure.cpu()
                if isinstance(p.structure, torch.Tensor) and p.structure.is_cuda
                else p.structure
            )
            sequence_tokens_list.append(seq_tok)
            structure_tokens_list.append(struct_tok)

        if use_structure:
            inputs_for_model = {
                "sequence_tokens": torch.stack(sequence_tokens_list),
                "structure_tokens": torch.stack(structure_tokens_list),
            }
        else:
            inputs_for_model = {
                "sequence_tokens": torch.stack(sequence_tokens_list),
            }
        return {k: v for k, v in inputs_for_model.items() if v is not None}

    @staticmethod
    def _esm3_transformer_hidden_states(
        model: ESM3, inputs: dict
    ) -> tuple[torch.Tensor, ...]:
        """Run ESM3 encoder + transformer and return per-block hidden states."""
        from esm.utils.constants import esm3 as C
        from esm.utils.structure.affine3d import build_affine3d_from_coordinates

        sequence_tokens = inputs.get("sequence_tokens")
        structure_tokens = inputs.get("structure_tokens")
        ss8_tokens = inputs.get("ss8_tokens")
        sasa_tokens = inputs.get("sasa_tokens")
        function_tokens = inputs.get("function_tokens")
        residue_annotation_tokens = inputs.get("residue_annotation_tokens")
        average_plddt = inputs.get("average_plddt")
        per_res_plddt = inputs.get("per_res_plddt")
        structure_coords = inputs.get("structure_coords")
        chain_id = inputs.get("chain_id")
        sequence_id = inputs.get("sequence_id")

        try:
            L, device = next(
                (x.shape[1], x.device)
                for x in [
                    sequence_tokens,
                    structure_tokens,
                    ss8_tokens,
                    sasa_tokens,
                    structure_coords,
                    function_tokens,
                    residue_annotation_tokens,
                ]
                if x is not None
            )
        except StopIteration:
            raise ValueError("At least one of the inputs must be non-None")

        batch_size = sequence_tokens.shape[0] if sequence_tokens is not None else 1
        t = model.tokenizers

        def defaults(x, tok):
            if x is None:
                return torch.full((batch_size, L), tok, dtype=torch.long, device=device)
            return x

        sequence_tokens = defaults(sequence_tokens, t.sequence.mask_token_id)
        ss8_tokens = defaults(ss8_tokens, C.SS8_PAD_TOKEN)
        sasa_tokens = defaults(sasa_tokens, C.SASA_PAD_TOKEN)
        average_plddt = defaults(average_plddt, 1).float()
        per_res_plddt = defaults(per_res_plddt, 0).float()
        chain_id = defaults(chain_id, 0)

        if residue_annotation_tokens is None:
            residue_annotation_tokens = torch.full(
                (batch_size, L, 16),
                C.RESIDUE_PAD_TOKEN,
                dtype=torch.long,
                device=device,
            )

        if function_tokens is None:
            function_tokens = torch.full(
                (batch_size, L, 8),
                C.INTERPRO_PAD_TOKEN,
                dtype=torch.long,
                device=device,
            )

        if structure_coords is None:
            structure_coords = torch.full(
                (batch_size, L, 3, 3), float("nan"), dtype=torch.float, device=device
            )

        structure_coords = structure_coords[..., :3, :]
        affine, affine_mask = build_affine3d_from_coordinates(structure_coords)

        structure_tokens = defaults(structure_tokens, C.STRUCTURE_MASK_TOKEN)
        assert structure_tokens is not None
        structure_tokens = (
            structure_tokens.masked_fill(structure_tokens == -1, C.STRUCTURE_MASK_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_BOS_TOKEN, C.STRUCTURE_BOS_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_PAD_TOKEN, C.STRUCTURE_PAD_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_EOS_TOKEN, C.STRUCTURE_EOS_TOKEN)
            .masked_fill(
                sequence_tokens == C.SEQUENCE_CHAINBREAK_TOKEN,
                C.STRUCTURE_CHAINBREAK_TOKEN,
            )
        )

        x = model.encoder(
            sequence_tokens,
            structure_tokens,
            average_plddt,
            per_res_plddt,
            ss8_tokens,
            sasa_tokens,
            function_tokens,
            residue_annotation_tokens,
        )
        _, _, hidden_states, _ = model.transformer(
            x,
            sequence_id,
            affine,
            affine_mask,
            chain_id,
            output_attentions=False,
        )
        return hidden_states

    @torch.no_grad()
    def forward(self, input):
        """
        Generate embeddings for the input sequences.

        Runs the ESM3 encoder + transformer via the compute core. Yields per-sequence
        embeddings (BOS/EOS stripped). Supports sequence-only or sequence+structure batches.
        Args:
            input: Dict from prepare_sequences with "sequence_tokens" and
                optional "structure_tokens" (same-length batch, no padding).

        Yields:
            Embeddings for each sequence (per-residue, without special tokens).
        """
        hidden = self._run_core(_esm3_core_inputs(input), (self.repr_layer,))
        for i in range(hidden.shape[0]):
            yield hidden[i, 0, 1:-1].cpu()

    @torch.no_grad()
    def forward_multi(self, input, repr_layers: list[int]):
        hidden = self._run_core(_esm3_core_inputs(input), repr_layers)
        for i in range(hidden.shape[0]):
            for j, layer in enumerate(repr_layers):
                yield i, layer, hidden[i, j, 1:-1].cpu()


def _esm3_core_inputs(input) -> dict:
    inputs = {"sequence_tokens": input["sequence_tokens"]}
    if input.get("structure_tokens") is not None:
        inputs["structure_tokens"] = input["structure_tokens"]
    return inputs


def _esm3_core(
    backbone: ESM3, inputs: dict, layers: tuple[int, ...], use_norm_layer: bool
):
    device = inputs["sequence_tokens"].device
    # ESM3 is often bfloat16; autocast avoids Float/BFloat16 mismatch. Entered here
    # (not by the caller) because autocast state is thread-local per DataParallel replica.
    with (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    ):
        hidden_states = ESM3EmbeddingModel._esm3_transformer_hidden_states(
            backbone, inputs
        )
    num_layers = len(hidden_states)

    selected = []
    for layer in layers:
        embeddings = _esm3_hidden_at_layer(hidden_states, layer)
        if use_norm_layer and layer == num_layers:
            embeddings = backbone.transformer.norm(embeddings)
        selected.append(embeddings.float())
    return torch.stack(selected, dim=1)


class ESMCEmbeddingModel(BaseProteinEmbeddingModel):
    embedding_kind = EmbeddingType.PER_RESIDUE
    chain_break_token: str = "|"

    def __init__(
        self,
        model_name: str,
        repr_layer: int | None = None,
        use_norm_layer: bool = True,
    ):
        super().__init__()
        num_layers = _esmc_num_layers_from_name(model_name)
        self.repr_layer = resolve_repr_layer(repr_layer, num_layers)
        validate_repr_layer(self.repr_layer, num_layers, model_name=model_name)

        self.model, self.tokenizer = load_esmc(model_name)
        self.pad_idx = self.tokenizer.pad_token_id
        _assert_esmc_num_layers(self.model, num_layers, model_name=model_name)
        self.use_norm_layer = use_norm_layer
        self._set_core(_esmc_core, use_norm_layer=use_norm_layer)

    def prepare_sequences(self, sequences, structures=None):
        return tokenize_esmc_sequences(self.model, sequences)

    def _attention_mask(self, inputs: dict, device: torch.device) -> torch.Tensor:
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = inputs["input_ids"] != self.pad_idx
        return attention_mask.to(device).bool()

    @torch.no_grad()
    def forward(self, input):
        inputs = dict(input) if hasattr(input, "keys") else {"input_ids": input}
        hidden = self._run_core(inputs, (self.repr_layer,))
        attention_mask = self._attention_mask(inputs, hidden.device)

        for i in range(hidden.shape[0]):
            x = hidden[i, 0][attention_mask[i]]
            yield x[1:-1].cpu()

    @torch.no_grad()
    def forward_multi(self, input, repr_layers: list[int]):
        inputs = dict(input) if hasattr(input, "keys") else {"input_ids": input}
        hidden = self._run_core(inputs, repr_layers)
        attention_mask = self._attention_mask(inputs, hidden.device)

        for j, layer in enumerate(repr_layers):
            for i in range(hidden.shape[0]):
                x = hidden[i, j][attention_mask[i]]
                yield i, layer, x[1:-1].cpu()


class CarpEmbeddingModel(BaseProteinEmbeddingModel):
    embedding_kind = EmbeddingType.PER_PROTEIN
    # CARP's PROTEIN_ALPHABET is "ACDEFGHIKLMNPQRSTVWYBZXJOU*-#@" and holds no
    # chain-break character: sequence_models defines SEP as "/", but only inside
    # MSA_ALPHABET_PLUS, so the collater raises KeyError on the canonical "|".
    # GAP is the one alphabet member with separator-like semantics, so multi-chain
    # input joins on it. Declaring the token here is enough for
    # EmbeddingManager._normalize_chain_breaks to rewrite "|" before tokenization,
    # and cache keys canonicalize back to "|" before hashing, so they do not move.
    chain_break_token: str = GAP

    def __init__(self, model_name: str, repr_layer: int | None = None):
        super().__init__()

        self.model, self.collater = load_model_and_alphabet(model_name)
        self.model.eval()
        # CARP wraps a ByteNetLM and exposes no layer count of its own, so take it from
        # the ByteNet block list.
        num_layers = len(self.model.model.embedder.layers)  # pyright: ignore[reportArgumentType]
        self.repr_layer = resolve_repr_layer(repr_layer, num_layers)
        validate_repr_layer(self.repr_layer, num_layers, model_name=model_name)
        self._set_core(_carp_core)

    def prepare_sequences(self, sequences, structures=None):
        logger.debug(f"Sequences: {sequences}")
        lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
        sequences = [[s] for s in sequences]

        batch_tokens = self.collater(sequences)[0]  # pyright: ignore[reportIndexIssue]
        logger.debug(f"batch: {batch_tokens}")

        return {"tokens": batch_tokens, "lengths": lengths}

    def _sequence_lengths(self, input) -> torch.Tensor:
        """Unpadded length per row.

        GAP and PAD are both "-" in CARP, so a chain break is indistinguishable from
        padding once tokenized: counting non-pad tokens undercounts by one per break
        and the contiguous slice that follows would drop the final residue of every
        multi-chain sequence. Use the lengths recorded at tokenization time, and keep
        the token comparison only for inputs built without them.
        """
        lengths = input.get("lengths")
        if lengths is not None:
            return lengths
        return (input["tokens"] != self.collater.pad_idx).sum(1)  # pyright: ignore[reportAttributeAccessIssue]

    @torch.no_grad()
    def forward(self, input):
        logger.debug(f"Input: {input}")
        hidden = self._run_core(input, (self.repr_layer,))
        seq_lengths = self._sequence_lengths(input)
        logger.debug(f"Sequence lengths: {seq_lengths}")

        for i, seq_len in enumerate(seq_lengths):
            yield hidden[i, 0, :seq_len]

    @torch.no_grad()
    def forward_multi(self, input, repr_layers: list[int]):
        logger.debug(f"Input: {input}")
        hidden = self._run_core(input, repr_layers)
        seq_lengths = self._sequence_lengths(input)

        for i, seq_len in enumerate(seq_lengths):
            for j, layer in enumerate(repr_layers):
                yield i, layer, hidden[i, j, :seq_len]


class ProteinMPNNEmbeddingModel(BaseProteinEmbeddingModel):
    embedding_kind = EmbeddingType.PER_RESIDUE
    structure_input = StructureInput.REQUIRED
    # One backbone per batch, so every sequence has to match its length.
    requires_same_length_batch = True

    def __init__(
        self, model_name: str, ca_only: bool = False, use_structure: bool = True
    ):
        super().__init__()
        print(f"Initializing ProteinMPNN embedding model with model name: {model_name}")

        cache_dir = torch.hub.get_dir()
        if os.path.exists(
            os.path.join(cache_dir, "ProteinMPNN", "weights", model_name)
        ):
            model_name = os.path.join(cache_dir, "ProteinMPNN", "weights", model_name)
        else:
            model_name = os.path.join(os.path.dirname(__file__), "weights", model_name)
        print(f"Loading ProteinMPNN model from {model_name}")
        self.model = ProteinMPNN.from_pretrained(model_name, ca_only=ca_only)
        self.model.eval()
        self.ca_only = ca_only
        self._set_core(_mpnn_core)

    def prepare_sequences(
        self, sequences, structures=None, chain_list=None, structure_id=None
    ):
        """
        Prepare sequences and structures for ProteinMPNN embedding.

        Args:
            sequences: List of protein sequences. Chains are separated by ``|`` and
                follow ``chain_list`` (or the parser's chain order without it). Each
                chain has one character per residue present in the structure.
            structures: PDB file path (string) or list of PDB file paths (required)
            chain_list: Chains to use, in the order of the sequences. ``None`` takes
                every chain in the file.
            structure_id: Unused; accepted so callers can pass it to any embedder.

        Returns:
            Dictionary containing featurized inputs ready for forward pass. ``keep``
            holds, per sample, the rows of the padded layout that are residues: the
            parser's filler rows for missing residue numbers are left out.
        """
        logger.debug(f"Preparing {len(sequences)} sequences for ProteinMPNN")
        if structures is None:
            raise ValueError(
                "ProteinMPNN requires structure information (PDB files). "
                "Please provide structures parameter when using ProteinMPNN. "
                "ProteinMPNN is a structure-conditioned model and cannot work with sequences alone."
            )

        device = next(self.model.parameters()).device

        first_pdb = structures if isinstance(structures, str) else structures[0]
        pdb_dict_list = parse_PDB(
            first_pdb, input_chain_list=chain_list, ca_only=self.ca_only
        )
        logger.debug(f"pdb_dict_list: {len(pdb_dict_list)}")
        if not pdb_dict_list:
            raise ValueError(f"Failed to parse PDB file: {first_pdb}")

        pdb_dict = pdb_dict_list[0]
        layout = _mpnn_chain_layout(pdb_dict)

        if sequences:
            batch = [_mpnn_threaded(pdb_dict, layout, seq) for seq in sequences]
            layouts = [layout] * len(batch)
        elif not isinstance(structures, str) and structures:
            batch, layouts = [], []
            for pdb_path in structures:
                pl = parse_PDB(
                    pdb_path, input_chain_list=chain_list, ca_only=self.ca_only
                )
                if not pl:
                    raise ValueError(f"Failed to parse PDB file: {pdb_path}")
                batch.append(pl[0])
                layouts.append(_mpnn_chain_layout(pl[0]))
        else:
            batch, layouts = [pdb_dict], [layout]

        featurized = tied_featurize(
            batch,
            device,
            chain_dict=None,
            fixed_position_dict=None,
            omit_AA_dict=None,
            tied_positions_dict=None,
            pssm_dict=None,
            bias_by_res_dict=None,
            ca_only=self.ca_only,
        )

        return {
            "X": featurized[0],
            "S": featurized[1],
            "mask": featurized[2],
            "chain_M": featurized[4],
            "residue_idx": featurized[12],
            "chain_encoding_all": featurized[5],
            "keep": [torch.as_tensor(lay["keep"]) for lay in layouts],
        }

    @torch.no_grad()
    def forward(self, input):
        """
        Generate embeddings for the input sequences/structures.

        Args:
            input: Dictionary containing featurized inputs from prepare_sequences

        Yields:
            Embeddings for each sequence (per-residue)
        """
        core_inputs = {k: input[k] for k in _MPNN_CORE_KEYS}
        h_V = self._run_core(core_inputs, (0,))
        # ``keep`` stays out of the core: it is not a batch-first tensor to scatter.
        for i, keep in enumerate(input["keep"]):
            yield h_V[i, 0][keep.to(h_V.device)].cpu()


_MPNN_CORE_KEYS = ("X", "S", "mask", "residue_idx", "chain_encoding_all")


def _mpnn_chain_layout(pdb_dict: dict) -> dict:
    """Which rows of a parsed PDB are residues, and where they land in the batch.

    ``parse_PDB_biounits`` adds an all-NaN filler row (``-``) for every missing residue
    number. Upstream keeps them so ``residue_idx`` spacing shows the numbering gap, and
    so does this wrapper, but they are not residues: sequences built from the structure
    (haipr, ESM3) do not count them. ``present`` lists the real rows per chain, in
    parse order; ``keep`` gives the same rows in ``tied_featurize``'s layout, which sorts
    chains alphabetically, ordered like ``letters``.
    """
    letters = [k[len("seq_chain_") :] for k in pdb_dict if k.startswith("seq_chain_")]
    present = {}
    for letter in letters:
        seq = pdb_dict[f"seq_chain_{letter}"]
        ca = np.asarray(pdb_dict[f"coords_chain_{letter}"][f"CA_chain_{letter}"], float)
        no_coords = np.isnan(ca.reshape(len(seq), -1)).all(axis=1)
        filler = np.array([c == "-" for c in seq], dtype=bool) & no_coords
        present[letter] = np.flatnonzero(~filler)
    offset, start = {}, 0
    for letter in sorted(letters):
        offset[letter] = start
        start += len(pdb_dict[f"seq_chain_{letter}"])
    keep = np.concatenate([offset[letter] + present[letter] for letter in letters])
    return {"letters": letters, "present": present, "keep": keep}


def _mpnn_threaded(pdb_dict: dict, layout: dict, seq: str) -> dict:
    """Copy of ``pdb_dict`` carrying ``seq``: one character per present residue.

    Chains are separated by ``|`` and follow ``layout["letters"]``. Without a
    separator the sequence is cut into consecutive per-chain pieces. Filler rows get
    ``X``.
    """
    letters, present = layout["letters"], layout["present"]
    counts = [len(present[letter]) for letter in letters]
    parts = seq.split("|")
    if len(parts) == 1 and len(letters) > 1:
        if len(seq) != sum(counts):
            raise ValueError(
                f"Sequence has {len(seq)} residues but the structure has "
                f"{sum(counts)} across chains {letters} ({dict(zip(letters, counts))})"
            )
        cuts = np.cumsum([0] + counts)
        parts = [seq[cuts[i] : cuts[i + 1]] for i in range(len(letters))]
    elif len(parts) != len(letters):
        raise ValueError(
            f"Sequence has {len(parts)} chain segments but the structure has "
            f"{len(letters)} chains {letters}"
        )

    out = pdb_dict.copy()
    for letter, part, count in zip(letters, parts, counts):
        if len(part) != count:
            raise ValueError(
                f"Chain {letter}: sequence has {len(part)} residues but the "
                f"structure has {count} (numbering gaps excluded)"
            )
        rows = ["X"] * len(pdb_dict[f"seq_chain_{letter}"])
        for row, char in zip(present[letter], part):
            rows[row] = char
        out[f"seq_chain_{letter}"] = "".join(rows)
    out["seq"] = "".join(out[f"seq_chain_{letter}"] for letter in letters)
    return out


def _mpnn_encode(backbone: nn.Module, inputs: dict):
    """Structure-only encoder, as in ``ProteinMPNN.forward``: nodes start at zero."""
    mask = inputs["mask"]
    E, E_idx = backbone.features(
        inputs["X"], mask, inputs["residue_idx"], inputs["chain_encoding_all"]
    )
    h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
    h_E = backbone.W_e(E)

    mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
    mask_attend = mask.unsqueeze(-1) * mask_attend
    for layer in backbone.encoder_layers:
        h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
    return h_V, h_E, E_idx


def _mpnn_decode(backbone, inputs, h_V_enc, h_E, E_idx, visible):
    """Decoder node states with the sequence teacher-forced.

    ``visible[b, q, p]`` is 1 where position ``p`` is decoded before position ``q``,
    i.e. where ``q`` may see the identity of ``p`` (``order_mask_backward`` upstream).
    """
    mask = inputs["mask"]
    h_S = backbone.W_s(inputs["S"])
    h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
    h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)

    mask_attend = torch.gather(visible, 2, E_idx).unsqueeze(-1)
    mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
    mask_bw = mask_1D * mask_attend
    mask_fw = mask_1D * (1.0 - mask_attend)
    h_EXV_encoder_fw = mask_fw * h_EXV_encoder

    h_V = h_V_enc
    for layer in backbone.decoder_layers:
        h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
        h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
        h_V = layer(h_V, h_ESV, mask)
    return h_V


def _mpnn_core(backbone: nn.Module, inputs: dict, layers: tuple[int, ...]):
    """ProteinMPNN decoder state before ``W_out``, ``[B, 1, L, D]``, sequence unmasked.

    The encoder sees the backbone only; the sequence enters through the decoder, which
    gets every residue identity, the position's own included (a residue is its own
    nearest neighbor). One pass per batch, like the other embedders. ``layers`` is
    ignored (single output).

    Training never showed the decoder this input: each pair of positions saw one
    another in at most one direction, and no position saw itself. On hYAP65 the states
    still track the leave-one-out ones (per-position cosine 0.994) and do not copy the
    identity (``W_out`` recovers 26% of residues).
    """
    h_V_enc, h_E, E_idx = _mpnn_encode(backbone, inputs)
    B, L = inputs["S"].shape
    visible = torch.ones(B, L, L, device=h_V_enc.device)
    return _mpnn_decode(backbone, inputs, h_V_enc, h_E, E_idx, visible).unsqueeze(1)


@dataclass
class ModelCard:
    name: str
    family: str
    embed_dim: int
    init_fn: Callable[[], BaseProteinEmbeddingModel]
    num_layers: int | None = None
    supports_repr_layer: bool = True
    structure_input: StructureInput = StructureInput.NONE

    @classmethod
    def from_model_cls(cls, *, model_cls, model_kwargs, **kwargs):
        kwargs.setdefault("structure_input", model_cls.structure_input)

        def init_fn(**init_kwargs):
            return model_cls(**{**model_kwargs, **init_kwargs})

        return cls(init_fn=init_fn, **kwargs)


model_descriptions = [
    ModelCard.from_model_cls(
        name="carp",
        family="CARP",
        embed_dim=1280,
        num_layers=56,
        model_cls=CarpEmbeddingModel,
        model_kwargs=dict(model_name="carp_640M"),
    ),
    ModelCard.from_model_cls(
        name="prot_t5_xl_uniref50",
        family="ProtTrans",
        embed_dim=1024,
        model_cls=ProtT5EmbeddingModel,
        model_kwargs=dict(model_name="prot_t5_xl_uniref50"),
    ),
    ModelCard.from_model_cls(
        name="prot_t5_xl_bfd",
        family="ProtTrans",
        embed_dim=1024,
        model_cls=ProtT5EmbeddingModel,
        model_kwargs=dict(model_name="prot_t5_xl_bfd"),
    ),
    ModelCard.from_model_cls(
        name="prot_t5_xxl_uniref50",
        family="ProtTrans",
        embed_dim=1024,
        model_cls=ProtT5EmbeddingModel,
        model_kwargs=dict(model_name="prot_t5_xxl_uniref50"),
    ),
    ModelCard.from_model_cls(
        name="prot_t5_xxl_bfd",
        family="ProtTrans",
        embed_dim=1024,
        model_cls=ProtT5EmbeddingModel,
        model_kwargs=dict(model_name="prot_t5_xxl_bfd"),
    ),
    ModelCard.from_model_cls(
        name="prot_bert_bfd",
        family="ProtTrans",
        embed_dim=1024,
        model_cls=ProtBERTEmbeddingModel,
        model_kwargs=dict(model_name="prot_bert_bfd"),
    ),
    ModelCard.from_model_cls(
        name="prot_bert",
        family="ProtTrans",
        embed_dim=1024,
        model_cls=ProtBERTEmbeddingModel,
        model_kwargs=dict(model_name="prot_bert"),
    ),
    ModelCard.from_model_cls(
        name="esm2_t48",
        family="ESM",
        embed_dim=5120,
        num_layers=48,
        model_cls=ESMEmbeddingModel,
        model_kwargs=dict(model_name="esm2_t48_15B_UR50D"),
    ),
    ModelCard.from_model_cls(
        name="esm2_t36",
        family="ESM",
        embed_dim=2560,
        num_layers=36,
        model_cls=ESMEmbeddingModel,
        model_kwargs=dict(model_name="esm2_t36_3B_UR50D"),
    ),
    ModelCard.from_model_cls(
        name="esm2_t33",
        family="ESM",
        embed_dim=1280,
        num_layers=33,
        model_cls=ESMEmbeddingModel,
        model_kwargs=dict(model_name="esm2_t33_650M_UR50D"),
    ),
    ModelCard.from_model_cls(
        name="esm2_t30",
        family="ESM",
        embed_dim=640,
        num_layers=30,
        model_cls=ESMEmbeddingModel,
        model_kwargs=dict(model_name="esm2_t30_150M_UR50D"),
    ),
    ModelCard.from_model_cls(
        name="esm2_t12",
        family="ESM",
        embed_dim=480,
        num_layers=12,
        model_cls=ESMEmbeddingModel,
        model_kwargs=dict(model_name="esm2_t12_35M_UR50D"),
    ),
    ModelCard.from_model_cls(
        name="esm2_t6",
        family="ESM",
        embed_dim=320,
        num_layers=6,
        model_cls=ESMEmbeddingModel,
        model_kwargs=dict(model_name="esm2_t6_8M_UR50D"),
    ),
    ModelCard.from_model_cls(
        name="esmc_600m",
        family="ESM",
        embed_dim=1152,
        num_layers=36,
        model_cls=ESMCEmbeddingModel,
        model_kwargs=dict(model_name="esmc_600m"),
    ),
    ModelCard.from_model_cls(
        name="esmc_6b",
        family="ESM",
        embed_dim=2560,
        num_layers=80,
        model_cls=ESMCEmbeddingModel,
        model_kwargs=dict(model_name="esmc_6b"),
    ),
    ModelCard.from_model_cls(
        name="esmc_300m",
        family="ESM",
        embed_dim=960,
        num_layers=30,
        model_cls=ESMCEmbeddingModel,
        model_kwargs=dict(model_name="esmc_300m"),
    ),
    ModelCard.from_model_cls(
        name="esm3",
        family="ESM",
        embed_dim=1536,
        num_layers=48,
        model_cls=ESM3EmbeddingModel,
        model_kwargs=dict(model_name="esm3_sm_open_v1", use_norm_layer=True),
    ),
    ModelCard.from_model_cls(
        name="saprot_35m",
        family="SaProt",
        embed_dim=480,
        num_layers=12,
        model_cls=SaProtEmbeddingModel,
        model_kwargs=dict(model_name="westlake-repl/SaProt_35M_AF2", num_layers=12),
    ),
    ModelCard.from_model_cls(
        name="saprot_650m",
        family="SaProt",
        embed_dim=1280,
        num_layers=33,
        model_cls=SaProtEmbeddingModel,
        model_kwargs=dict(model_name="westlake-repl/SaProt_650M_AF2", num_layers=33),
    ),
    ModelCard.from_model_cls(
        name="saprot_650m_pdb",
        family="SaProt",
        embed_dim=1280,
        num_layers=33,
        model_cls=SaProtEmbeddingModel,
        model_kwargs=dict(model_name="westlake-repl/SaProt_650M_PDB", num_layers=33),
    ),
    ModelCard.from_model_cls(
        name="saprot_1_3b",
        family="SaProt",
        embed_dim=1280,
        num_layers=66,
        model_cls=SaProtEmbeddingModel,
        model_kwargs=dict(
            model_name="westlake-repl/SaProt_1.3B_AFDB_OMG_NCBI", num_layers=66
        ),
    ),
    ModelCard.from_model_cls(
        name="mpnn",
        family="ProteinMPNN",
        embed_dim=128,
        supports_repr_layer=False,
        model_cls=ProteinMPNNEmbeddingModel,
        model_kwargs=dict(model_name="v_48_020.pt", ca_only=True),
    ),
]


model_dict: dict[str, ModelCard] = OrderedDict((m.name, m) for m in model_descriptions)

model_families = set(m.family for m in model_descriptions)


def list_models(family: str | None = None):
    if family is not None:
        if family not in model_families:
            raise ValueError(
                f"Unknown model family '{family}'. Available families are {model_families}"
            )

        return [m.name for m in model_descriptions if m.family == family]
    else:
        return list(model_dict)


def get_model_info(model_name: str):
    if model_name not in model_dict:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models are {list_models()}"
        )

    model_desc = model_dict[model_name]

    return {
        "name": model_desc.name,
        "family": model_desc.family,
        "embed_dim": model_desc.embed_dim,
        "num_layers": model_desc.num_layers,
        "default_repr_layer": model_desc.num_layers,
        "supports_repr_layer": model_desc.supports_repr_layer,
        "structure_input": model_desc.structure_input.value,
    }


def get_model(model_name, repr_layer=None, repr_layers=None, **kwargs):
    if model_name not in model_dict:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models are {list_models()}"
        )

    model_desc = model_dict[model_name]
    init_kwargs = dict(kwargs)

    layers = None
    if model_desc.num_layers is not None:
        layers = parse_repr_layers_config(
            repr_layer, repr_layers, model_desc.num_layers, model_name=model_name
        )
        # Initialize the underlying model with the highest requested layer so that
        # the single-layer forward() path remains consistent with repr_layer.
        init_kwargs["repr_layer"] = layers[-1]
    elif repr_layer is not None:
        init_kwargs["repr_layer"] = repr_layer

    requests_repr_layer = repr_layer is not None or bool(repr_layers)
    if not model_desc.supports_repr_layer and requests_repr_layer:
        warnings.warn(
            f"Model '{model_name}' does not support repr_layer configuration."
        )
        raise ValueError(
            f"Model '{model_name}' does not support repr_layer configuration."
        )

    model = model_desc.init_fn(**init_kwargs)

    if layers is not None:
        model.repr_layers = layers

    return model
