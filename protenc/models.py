from dataclasses import dataclass
import torch
import torch.nn as nn
import attr
import contextlib
from collections import OrderedDict
from typing import Callable
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
from esm.models.esmc import ESMC
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, ESMProteinTensor
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils import encoding
from esm.utils.sampling import _BatchedESMProteinTensor
from esm.utils.generation import _batch_forward

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class EmbeddingType(Enum):
    PER_RESIDUE = "per_residue"
    PER_PROTEIN = "per_protein"


class BaseProteinEmbeddingModel(nn.Module):
    embedding_type: EmbeddingType

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

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        # Cache device to prevent issues with DataParallel
        try:
            self._device = next(self.model.parameters()).device
        except StopIteration:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _validate_model_name(self, model_name):
        assert (
            self.available_models is None or model_name in self.available_models
        ), f"Unknown model name '{model_name}'. Available options are {self.available_models}"

    def prepare_sequences(self, sequences):
        # ProtTrans tokenizers expect whitespaces between residues
        sequences = [" ".join(s.replace(" ", "")) for s in sequences]

        # Simply return the encoded sequences without TensorDict
        return self.tokenizer.batch_encode_plus(
            sequences, return_tensors="pt", add_special_tokens=True, padding=True
        )

    def _post_process_embedding(self, embed, seq_len):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        attn_mask = input["attention_mask"]

        output = self.model(**input)

        embeddings = output.last_hidden_state  # Don't move to CPU
        seq_lens = (attn_mask == 1).sum(-1)

        results = []
        for embed, seq_len in zip(embeddings, seq_lens):
            # Tokenized sequences have the following form:
            # [CLS] V N ... I K [SEP] [PAD] ... [PAD]
            #
            # We remove the special tokens ([CLS], [SEP], [PAD]) before
            # computing the mean over the remaining sequence
            results.append(self._post_process_embedding(embed, seq_len))

        return results


class ProtBERTEmbeddingModel(BaseProtTransEmbeddingModel):
    available_models = ["prot_bert", "prot_bert_bfd"]
    structure_aware = False

    def __init__(self, model_name: str, load_weights: bool = True):
        self._validate_model_name(model_name)

        mode_name = f"Rostlab/{model_name}"
        model, tokenizer = load_huggingface_language_model(
            BertModel, BertTokenizer, mode_name, load_weights=load_weights
        )

        super().__init__(model=model, tokenizer=tokenizer)

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

    def __init__(self, model_name: str, load_weights: bool = True):
        self._validate_model_name(model_name)

        mode_name = f"Rostlab/{model_name}"
        model, tokenizer = load_huggingface_language_model(
            T5EncoderModel, T5Tokenizer, mode_name, load_weights=load_weights
        )

        super().__init__(model=model, tokenizer=tokenizer)

    def _post_process_embedding(self, embedding, seq_len):
        return embedding[: seq_len - 1]


class ESMEmbeddingModel(BaseProteinEmbeddingModel):
    embedding_kind = EmbeddingType.PER_RESIDUE
    structure_aware = False

    def __init__(self, model_name: str, repr_layer: int):
        super().__init__()

        self.model = EsmModel.from_pretrained("facebook/" + model_name)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/" + model_name)

        self.model.eval()
        self.repr_layer = repr_layer
        # Cache device to prevent issues with DataParallel
        try:
            self._device = next(self.model.parameters()).device
        except StopIteration:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def clean(self, seq):
        if not re.match(r"^[ACDEFGHIKLMNPQRSTVWYX]+$", seq):
            print(f"Invalid sequence: {seq}")
            # convert unknown characters to X
            seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWYX]", "X", seq)
            print(f"Converted sequence: {seq}")
        return seq

    def prepare_sequences(self, sequences):
        batch_tokens = self.tokenizer(
            sequences, return_tensors="pt", add_special_tokens=True, padding=True
        )

        # Simply return a dictionary with the tokens
        return batch_tokens

    @torch.no_grad()
    def forward(self, input):
        logger.debug(f"Input: {input}")
        results = self.model(**input, output_hidden_states=False)
        token_representations = results["last_hidden_state"]
        seq_lengths = input["attention_mask"].sum(1)

        embeddings = []
        for i, seq_len in enumerate(seq_lengths):
            embeddings.append(token_representations[i, 1 : seq_len - 1])

        return embeddings


class ESM3EmbeddingModel(BaseProteinEmbeddingModel):
    # largely taken from https://github.com/Dianzhuo-Wang/esm3-structural-inputs/esm/
    embedding_kind = EmbeddingType.PER_RESIDUE
    structure_aware = True

    def __init__(self, model_name: str, use_norm_layer: bool = True):
        """
        Initialize the ESM3 embedding model.

        Args:
            model_name: Name of the ESM3 model to use
            use_norm_layer: Whether to use the normalized embeddings (transformer.norm applied)
                           or raw embeddings from the model
        """
        super().__init__()
        self.model: ESM3 = ESM3.from_pretrained(model_name)
        self.model.eval()
        self.use_norm_layer = use_norm_layer

    def prepare_sequences(self, sequences, structure_path: str | None = None):
        """
        Prepare sequences and structures for ESM3 embedding.

        Args:
            sequences: List of protein sequences
            structures: Optional path to a protein structure (PDB file)

        Returns:
            List of ESMProteinTensor objects ready for embedding
        """
        # Convert sequences to ESMProtein objects and encode them
        prots = [self.model.encode(ESMProtein(sequence=seq)) for seq in sequences]

        # Process structures if provided
        if structure_path is not None:

            # Use the structure for all sequences
            chain = ProteinChain.from_pdb(structure_path)
            protein = ESMProtein.from_protein_chain(chain)

            # Check same sequence length
            if len(protein.sequence) != len(sequences[0]):
                raise ValueError(
                    f"Sequence length mismatch between structure and sequence: {len(protein.sequence)} != {len(sequences[0])}"
                )

            # Process structure
            coords, _, struct_tokens = encoding.tokenize_structure(
                protein.coordinates,
                self.model.get_structure_encoder(),
                structure_tokenizer=self.model.tokenizers.structure,
                reference_sequence="",
                add_special_tokens=True,
            )

            # Apply structure to all proteins
            for prot in prots:
                prot.coordinates = coords
                prot.structure = struct_tokens

        return prots

    @torch.no_grad()
    def forward(self, input):
        """
        Generate embeddings for the input sequences.

        Args:
            input: Single ESMProteinTensor or list of ESMProteinTensor objects

        Returns:
            List of embeddings for each sequence (without special tokens)
        """
        results = []

        # Get device once at the beginning to avoid StopIteration error in DataParallel
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            # Fallback if no parameters - could happen in DataParallel context
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for protein_tensor in input:
            # Make a copy of the input
            protein_tensor = attr.evolve(protein_tensor)

            # Initialize default values for missing tracks
            default_protein_tensor = ESMProteinTensor.empty(
                len(protein_tensor) - 2,
                tokenizers=self.model.tokenizers,
                device=protein_tensor.device,
            )
            for track in attr.fields(ESMProteinTensor):
                if getattr(protein_tensor, track.name, None) is None:
                    setattr(
                        protein_tensor,
                        track.name,
                        getattr(default_protein_tensor, track.name, None),
                    )

            if len(protein_tensor) <= 0:
                raise ValueError("No input data provided")

            # Move input protein to proper device
            batched_protein = _BatchedESMProteinTensor.from_protein_tensor(
                protein_tensor
            )
            batched_protein.to(device)

            # Get forward output
            forward_output = _batch_forward(self.model, batched_protein)

            # Apply layer norm if requested
            if self.use_norm_layer:
                with (
                    torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16, enabled=True
                    )
                    if device.type == "cuda"
                    else contextlib.nullcontext()
                ):
                    embeddings = self.model.transformer.norm(forward_output.embeddings)
            else:
                embeddings = forward_output.embeddings

            # Collect all embeddings without moving to CPU
            # Let DataParallel handle device movement
            for e in embeddings:
                # Remove special tokens (first and last)
                results.append(e[1:-1])

        return results


class ESMCEmbeddingModel(BaseProteinEmbeddingModel):
    embedding_kind = EmbeddingType.PER_RESIDUE
    structure_aware = False

    def __init__(self, model_name: str):
        super().__init__()
        self.model: ESMC = ESMC.from_pretrained(model_name)
        self.model.eval()
        self.pad_idx = self.model.tokenizer.pad_token_id
        # Cache device to prevent issues with DataParallel
        try:
            self._device = next(self.model.parameters()).device
        except StopIteration:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_sequences(self, sequences):
        input_ids = self.model._tokenize(sequences)
        self.padding_mask = input_ids != self.pad_idx
        return input_ids

    @torch.no_grad()
    def forward(self, input):
        output = self.model(input)
        results = []

        # Make sure padding_mask is on the same device as the tensors
        device = output.embeddings.device
        padding_mask = self.padding_mask.to(device)

        for i in range(len(output.embeddings)):
            x = output.embeddings[i]  # get embedding for sequence i
            x = x[padding_mask[i]]  # remove padding tokens
            results.append(x[1:-1])  # remove start and end tokens, keep on GPU

        return results


class CarpEmbeddingModel(BaseProteinEmbeddingModel):
    """
    CARP model wrapper. Return a per-sequence embedding.
    """

    embedding_kind = EmbeddingType.PER_PROTEIN
    structure_aware = False

    def __init__(self, model_name: str, repr_layer: int):
        super().__init__()
        # SimpleCollater for carp models
        self.model, self.collater = load_model_and_alphabet(model_name)
        self.model.eval()
        self.repr_layer = repr_layer
        # Cache device to prevent issues with DataParallel
        try:
            self._device = next(self.model.parameters()).device
        except StopIteration:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_sequences(self, sequences):
        logger.debug(f"Sequences: {sequences}")
        sequences = [
            [s] for s in sequences
        ]  # convert to list of lists otherwise collater will fail
        # returns (sequences,)
        batch_tokens = self.collater(sequences)[0]
        logger.debug(f"batch: {batch_tokens}")

        # Simply return a dictionary with the tokens
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

        # Check if per-protein (already aggregated) or per-residue
        if self.embedding_kind == EmbeddingType.PER_PROTEIN:
            # Just return the per-protein embeddings without CPU transfer
            return [token_representations[i] for i in range(len(seq_lengths))]
        else:
            # For per-residue, handle padding to ensure uniform size
            max_len = max(seq_lengths).item()
            embeddings = []
            embed_dim = token_representations.shape[-1]
            device = token_representations.device

            for i, seq_len in enumerate(seq_lengths):
                # Get the actual sequence embedding without CPU transfer
                seq_embed = token_representations[i, :seq_len]

                # Create padded version on the same device to ensure uniform shape
                padded_embed = torch.zeros((max_len, embed_dim), device=device)
                padded_embed[:seq_len] = seq_embed

                embeddings.append(padded_embed)

            return embeddings


@dataclass
class ModelCard:
    name: str
    family: str
    embed_dim: int
    init_fn: Callable[[], BaseProteinEmbeddingModel]

    @classmethod
    def from_model_cls(cls, *, model_cls, model_kwargs, **kwargs):
        def init_fn(**init_kwargs):
            return model_cls(**{**model_kwargs, **init_kwargs})

        return cls(init_fn=init_fn, **kwargs)


model_descriptions = [
    # CARP family (https://github.com/microsoft/protein-sequence-models)
    ModelCard.from_model_cls(
        name="carp_640M",
        family="CARP",
        embed_dim=1280,
        model_cls=CarpEmbeddingModel,
        model_kwargs=dict(model_name="carp_640M", repr_layer=56),
    ),
    # ProtTrans family (https://github.com/agemagician/ProtTrans)
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
    # ESM family (https://github.com/facebookresearch/esm)
    ModelCard.from_model_cls(
        name="esm2_t48",
        family="ESM",
        embed_dim=5120,
        model_cls=ESMEmbeddingModel,
        model_kwargs=dict(model_name="esm2_t48_15B_UR50D", repr_layer=48),
    ),
    ModelCard.from_model_cls(
        name="esm2_t36",
        family="ESM",
        embed_dim=2560,
        model_cls=ESMEmbeddingModel,
        model_kwargs=dict(model_name="esm2_t36_3B_UR50D", repr_layer=36),
    ),
    ModelCard.from_model_cls(
        name="esm2_t33",
        family="ESM",
        embed_dim=1280,
        model_cls=ESMEmbeddingModel,
        model_kwargs=dict(model_name="esm2_t33_650M_UR50D", repr_layer=33),
    ),
    ModelCard.from_model_cls(
        name="esm2_t30",
        family="ESM",
        embed_dim=640,
        model_cls=ESMEmbeddingModel,
        model_kwargs=dict(model_name="esm2_t30_150M_UR50D", repr_layer=30),
    ),
    ModelCard.from_model_cls(
        name="esm2_t12",
        family="ESM",
        embed_dim=480,
        model_cls=ESMEmbeddingModel,
        model_kwargs=dict(model_name="esm2_t12_35M_UR50D", repr_layer=12),
    ),
    ModelCard.from_model_cls(
        name="esm2_t6",
        family="ESM",
        embed_dim=320,
        model_cls=ESMEmbeddingModel,
        model_kwargs=dict(model_name="esm2_t6_8M_UR50D", repr_layer=6),
    ),
    # https://github.com/evolutionaryscale/esm/tree/main
    ModelCard.from_model_cls(
        name="esmc_600m",
        family="ESM",
        embed_dim=1152,
        model_cls=ESMCEmbeddingModel,
        model_kwargs=dict(model_name="esmc_600m"),
    ),
    ModelCard.from_model_cls(
        name="esmc_300m",
        family="ESM",
        embed_dim=960,
        model_cls=ESMCEmbeddingModel,
        model_kwargs=dict(model_name="esmc_300m"),
    ),
    ModelCard.from_model_cls(
        name="esm3",
        family="ESM",
        embed_dim=1536,
        model_cls=ESM3EmbeddingModel,
        model_kwargs=dict(model_name="esm3_sm_open_v1", use_norm_layer=True),
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
    }


def get_model(model_name, **kwargs):
    model = model_dict[model_name].init_fn(**kwargs)

    return model
