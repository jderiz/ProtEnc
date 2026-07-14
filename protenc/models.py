from dataclasses import dataclass
import contextlib
import os
import warnings
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Callable, List, Optional
from enum import Enum
from transformers import (
    BertModel,
    BertTokenizer,
    T5EncoderModel,
    T5Tokenizer,
    EsmModel,
    AutoTokenizer,
)
from sequence_models.pretrained import load_model_and_alphabet
import colorlog as logging
import re
from haipr.models.esmc_loading import (
    ESMC_HF_REPOS,
    esmc_hidden_at_layer,
    esmc_hidden_states,
    esmc_select_hidden_layer,
    hf_config_num_layers,
    load_esmc,
    tokenize_esmc_sequences,
)
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, ProteinComplex
from esm.utils.structure.protein_chain import ProteinChain

try:
    from .mpnn import ProteinMPNN, parse_PDB, tied_featurize, gather_nodes
except ImportError:
    from mpnn import ProteinMPNN, parse_PDB, tied_featurize, gather_nodes

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
    return type(path)(p.with_name(new_name)) if isinstance(path, str) else p.with_name(
        new_name
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


def _assert_hf_num_hidden_layers(
    model: nn.Module, expected: int, *, model_name: str
) -> None:
    actual = hf_config_num_layers(model.config)
    if actual is None:
        raise ValueError(
            f"Could not read layer count from config for '{model_name}'."
        )
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


class BaseProteinEmbeddingModel(nn.Module):
    embedding_type: EmbeddingType
    # Default: no explicit chain break token. Specific models may override.
    chain_break_token: str = ""
    # Layers to extract in the multi-layer path; set by get_model when configured.
    repr_layers: list[int] | None = None

    def prepare_sequences(self, sequences, structures=None):
        return NotImplementedError

    def forward(self, input):
        raise NotImplementedError


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
        attn_mask = input["attention_mask"]

        output = self.model(**input, output_hidden_states=True)
        hidden_states = output.hidden_states
        seq_lens = (attn_mask == 1).sum(-1)

        for i, seq_len in enumerate(seq_lens):
            embed = _hf_hidden_at_layer(hidden_states, self.repr_layer)[i]
            yield self._post_process_embedding(embed.cpu(), seq_len)

    @torch.no_grad()
    def forward_multi(self, input, repr_layers: list[int]):
        attn_mask = input["attention_mask"]

        output = self.model(**input, output_hidden_states=True)
        hidden_states = output.hidden_states
        seq_lens = (attn_mask == 1).sum(-1)

        for i, seq_len in enumerate(seq_lens):
            for layer in repr_layers:
                embed = _hf_hidden_at_layer(hidden_states, layer)[i]
                yield i, layer, self._post_process_embedding(embed.cpu(), seq_len)


class ProtBERTEmbeddingModel(BaseProtTransEmbeddingModel):
    available_models = ["prot_bert", "prot_bert_bfd"]
    structure_aware = False

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
    structure_aware = False

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

    def _post_process_embedding(self, embedding, seq_len):
        return embedding[: seq_len - 1]


class ESMEmbeddingModel(BaseProteinEmbeddingModel):
    embedding_kind = EmbeddingType.PER_RESIDUE
    structure_aware = False
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
        results = self.model(**input, output_hidden_states=True)
        token_representations = _hf_hidden_at_layer(
            results.hidden_states, self.repr_layer
        )

        seq_lengths = input["attention_mask"].sum(1)

        for i, seq_len in enumerate(seq_lengths):
            yield token_representations[i, 1 : seq_len - 1]

    @torch.no_grad()
    def forward_multi(self, input, repr_layers: list[int]):
        logger.debug(f"Input: {input}")
        results = self.model(**input, output_hidden_states=True)
        seq_lengths = input["attention_mask"].sum(1)

        for i, seq_len in enumerate(seq_lengths):
            for layer in repr_layers:
                token_representations = _hf_hidden_at_layer(
                    results.hidden_states, layer
                )
                yield i, layer, token_representations[i, 1 : seq_len - 1]


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
    structure_aware = True
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

    def _get_model(self):
        return (
            self.model.module
            if isinstance(self.model, torch.nn.DataParallel)
            else self.model
        )

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
        logger.info(f"Preparing sequences for ESM3")
        structure_path = (
            structures
            if isinstance(structures, str)
            else (structures[0] if structures else None)
        )
        use_structure = structure_path is not None
        model = self._get_model()

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
                    # order (split on "|"). Strip "|" to produce a flat sequence whose
                    # residue count matches shared_coordinates exactly.
                    flat_sequences = [seq.replace("|", "") for seq in sequences]
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
                                flat_sequences.append("".join(aligned_parts))
                            elif len(parts) == len(loaded_chain_ids):
                                flat_sequences.append(seq.replace("|", ""))
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

    def _esm3_transformer_hidden_states(
        self, model: ESM3, inputs: dict
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
                return torch.full(
                    (batch_size, L), tok, dtype=torch.long, device=device
                )
            return x
        sequence_tokens = defaults(sequence_tokens, t.sequence.mask_token_id)
        ss8_tokens = defaults(ss8_tokens, C.SS8_PAD_TOKEN)
        sasa_tokens = defaults(sasa_tokens, C.SASA_PAD_TOKEN)
        average_plddt = defaults(average_plddt, 1).float()
        per_res_plddt = defaults(per_res_plddt, 0).float()
        chain_id = defaults(chain_id, 0)

        if residue_annotation_tokens is None:
            residue_annotation_tokens = torch.full(
                (batch_size, L, 16), C.RESIDUE_PAD_TOKEN, dtype=torch.long, device=device
            )

        if function_tokens is None:
            function_tokens = torch.full(
                (batch_size, L, 8), C.INTERPRO_PAD_TOKEN, dtype=torch.long, device=device
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

        Uses model(**inputs) then output.embeddings. Yields per-sequence embeddings
        (BOS/EOS stripped). Supports sequence-only or sequence+structure batches.
        Args:
            input: Dict from prepare_sequences with "sequence_tokens" and
                optional "structure_tokens" (same-length batch, no padding).

        Yields:
            Embeddings for each sequence (per-residue, without special tokens).
        """
        device = next(self.model.parameters()).device
        model = self._get_model()

        inputs = {"sequence_tokens": input["sequence_tokens"].to(device)}
        if "structure_tokens" in input and input["structure_tokens"] is not None:
            inputs["structure_tokens"] = input["structure_tokens"].to(device)
        inputs = {k: v for k, v in inputs.items() if v is not None}

        # ESM3 is often bfloat16; run forward under autocast to avoid Float/BFloat16 mismatch
        with (
            torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
            if device.type == "cuda"
            else contextlib.nullcontext()
        ):
            hidden_states = self._esm3_transformer_hidden_states(model, inputs)
        num_layers = len(hidden_states)
        embeddings = _esm3_hidden_at_layer(hidden_states, self.repr_layer)
        if self.use_norm_layer and self.repr_layer == num_layers:
            embeddings = model.transformer.norm(embeddings)
        if embeddings.dtype == torch.bfloat16:
            embeddings = embeddings.float()

        batch_size = embeddings.shape[0]
        for i in range(batch_size):
            yield embeddings[i, 1:-1].cpu()

    @torch.no_grad()
    def forward_multi(self, input, repr_layers: list[int]):
        device = next(self.model.parameters()).device
        model = self._get_model()

        inputs = {"sequence_tokens": input["sequence_tokens"].to(device)}
        if "structure_tokens" in input and input["structure_tokens"] is not None:
            inputs["structure_tokens"] = input["structure_tokens"].to(device)
        inputs = {k: v for k, v in inputs.items() if v is not None}

        with (
            torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
            if device.type == "cuda"
            else contextlib.nullcontext()
        ):
            hidden_states = self._esm3_transformer_hidden_states(model, inputs)
        num_layers = len(hidden_states)

        for layer in repr_layers:
            embeddings = _esm3_hidden_at_layer(hidden_states, layer)
            if self.use_norm_layer and layer == num_layers:
                embeddings = model.transformer.norm(embeddings)
            if embeddings.dtype == torch.bfloat16:
                embeddings = embeddings.float()

            batch_size = embeddings.shape[0]
            for i in range(batch_size):
                yield i, layer, embeddings[i, 1:-1].cpu()


class ESMCEmbeddingModel(BaseProteinEmbeddingModel):
    embedding_kind = EmbeddingType.PER_RESIDUE
    structure_aware = False
    chain_break_token: str = "|"

    def __init__(self, model_name: str, repr_layer: int | None = None):
        super().__init__()
        num_layers = _esmc_num_layers_from_name(model_name)
        self.repr_layer = resolve_repr_layer(repr_layer, num_layers)
        validate_repr_layer(self.repr_layer, num_layers, model_name=model_name)

        self.model, self.tokenizer = load_esmc(model_name)
        self.pad_idx = self.tokenizer.pad_token_id
        _assert_hf_num_hidden_layers(self.model, num_layers, model_name=model_name)

    def prepare_sequences(self, sequences, structures=None):
        return tokenize_esmc_sequences(self.tokenizer, sequences)

    @torch.no_grad()
    def forward(self, input):
        inputs = dict(input) if hasattr(input, "keys") else {"input_ids": input}
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = inputs["input_ids"] != self.pad_idx
        embeddings = esmc_hidden_at_layer(self.model, inputs, self.repr_layer)

        for i in range(embeddings.shape[0]):
            x = embeddings[i][attention_mask[i].bool()]
            yield x[1:-1].cpu()

    @torch.no_grad()
    def forward_multi(self, input, repr_layers: list[int]):
        inputs = dict(input) if hasattr(input, "keys") else {"input_ids": input}
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = inputs["input_ids"] != self.pad_idx
        hidden_states = esmc_hidden_states(self.model, inputs)

        for layer in repr_layers:
            embeddings = esmc_select_hidden_layer(hidden_states, layer)
            for i in range(embeddings.shape[0]):
                x = embeddings[i][attention_mask[i].bool()]
                yield i, layer, x[1:-1].cpu()


class CarpEmbeddingModel(BaseProteinEmbeddingModel):
    embedding_kind = EmbeddingType.PER_PROTEIN
    structure_aware = False

    def __init__(self, model_name: str, repr_layer: int | None = None):
        super().__init__()

        self.model, self.collater = load_model_and_alphabet(model_name)
        self.model.eval()
        num_layers = int(self.model.num_layers)
        self.repr_layer = resolve_repr_layer(repr_layer, num_layers)
        validate_repr_layer(self.repr_layer, num_layers, model_name=model_name)

    def prepare_sequences(self, sequences, structures=None):
        logger.debug(f"Sequences: {sequences}")
        sequences = [[s] for s in sequences]

        batch_tokens = self.collater(sequences)[0]
        logger.debug(f"batch: {batch_tokens}")

        return {"tokens": batch_tokens}

    @torch.no_grad()
    def forward(self, input):
        logger.debug(f"Input: {input}")
        tokens = input["tokens"]
        results = self.model(tokens, repr_layers=[self.repr_layer], logits=False)
        token_representations = results["representations"][self.repr_layer]
        seq_lengths = (input["tokens"] != self.collater.pad_idx).sum(1)
        logger.debug(f"Token representations: {token_representations}")
        logger.debug(f"Sequence lengths: {seq_lengths}")

        for i, seq_len in enumerate(seq_lengths):
            yield token_representations[i, :seq_len]

    @torch.no_grad()
    def forward_multi(self, input, repr_layers: list[int]):
        logger.debug(f"Input: {input}")
        tokens = input["tokens"]
        results = self.model(tokens, repr_layers=list(repr_layers), logits=False)
        seq_lengths = (input["tokens"] != self.collater.pad_idx).sum(1)

        for i, seq_len in enumerate(seq_lengths):
            for layer in repr_layers:
                token_representations = results["representations"][layer]
                yield i, layer, token_representations[i, :seq_len]


class ProteinMPNNEmbeddingModel(BaseProteinEmbeddingModel):
    embedding_kind = EmbeddingType.PER_RESIDUE
    structure_aware = True

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

    def prepare_sequences(self, sequences, structures=None):
        """
        Prepare sequences and structures for ProteinMPNN embedding.

        Args:
            sequences: List of protein sequences
            structures: PDB file path (string) or list of PDB file paths (required)

        Returns:
            Dictionary containing featurized inputs ready for forward pass
        """
        logger.debug(f"Preparing {len(sequences)} sequences for ProteinMPNN")
        if structures is None:
            raise ValueError(
                "ProteinMPNN requires structure information (PDB files). "
                "Please provide structures parameter when using ProteinMPNN. "
                "ProteinMPNN is a structure-conditioned model and cannot work with sequences alone."
            )

        device = next(self.model.parameters()).device

        if isinstance(structures, str):
            pdb_dict_list = parse_PDB(structures, ca_only=self.ca_only)
        else:
            # List of PDB paths: use first for structure, batch from all if no sequences
            pdb_dict_list = (
                parse_PDB(structures[0], ca_only=self.ca_only) if structures else []
            )
        logger.debug(f"pdb_dict_list: {len(pdb_dict_list)}")
        if not pdb_dict_list:
            raise ValueError(
                f"Failed to parse PDB file: {structures if isinstance(structures, str) else structures[0] if structures else None}"
            )

        pdb_dict = pdb_dict_list[0]
        pdb_seq_len = len(pdb_dict.get("seq", ""))

        chain_keys = [key for key in pdb_dict.keys() if key.startswith("seq_chain_")]
        chain_letters = [key.replace("seq_chain_", "") for key in chain_keys]

        if sequences:
            if not all(len(seq) == pdb_seq_len for seq in sequences):
                raise ValueError(
                    f"All sequences ({len(sequences[0])}) must have the same length as PDB structure ({pdb_seq_len})"
                )
            batch = []
            for seq in sequences:
                seq_dict = pdb_dict.copy()
                seq_dict["seq"] = seq
                if len(chain_letters) == 1:
                    seq_dict[f"seq_chain_{chain_letters[0]}"] = seq
                else:
                    start_idx = 0
                    for letter in chain_letters:
                        chain_key = f"seq_chain_{letter}"
                        if chain_key in pdb_dict:
                            chain_len = len(pdb_dict[chain_key])
                            seq_dict[chain_key] = seq[start_idx : start_idx + chain_len]
                            start_idx += chain_len
                batch.append(seq_dict)
        elif not isinstance(structures, str) and structures:
            batch = []
            for pdb_path in structures:
                pl = parse_PDB(pdb_path, ca_only=self.ca_only)
                if not pl:
                    raise ValueError(f"Failed to parse PDB file: {pdb_path}")
                batch.append(pl[0])
        else:
            batch = [pdb_dict]

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
            "lengths": featurized[3],
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
        device = next(self.model.parameters()).device
        X = input["X"].to(device)
        S = input["S"].to(device)
        mask = input["mask"].to(device)
        residue_idx = input["residue_idx"].to(device)
        chain_encoding_all = input["chain_encoding_all"].to(device)
        lengths = input["lengths"]

        E, E_idx = self.model.features(X, mask, residue_idx, chain_encoding_all)
        h_S = self.model.W_s(S)
        h_V = h_S.clone()
        h_E = self.model.W_e(E)

        # Masking for attention (gather_nodes is used both in mpnn.py and here)
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        # Pass through encoder layers (exact order/inputs as in mpnn.py)
        for layer in self.model.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        for i, seq_len in enumerate(lengths):
            yield h_V[i, :seq_len].cpu()


@dataclass
class ModelCard:
    name: str
    family: str
    embed_dim: int
    init_fn: Callable[[], BaseProteinEmbeddingModel]
    num_layers: int | None = None
    supports_repr_layer: bool = True

    @classmethod
    def from_model_cls(cls, *, model_cls, model_kwargs, **kwargs):
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
