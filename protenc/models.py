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
from esm.models.esmc import ESMC, ESMCOutput
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

        embeddings = output.last_hidden_state.cpu()
        seq_lens = (attn_mask == 1).sum(-1)

        for embed, seq_len in zip(embeddings, seq_lens):
            # Tokenized sequences have the following form:
            # [CLS] V N ... I K [SEP] [PAD] ... [PAD]
            #
            # We remove the special tokens ([CLS], [SEP], [PAD]) before
            # computing the mean over the remaining sequence

            yield self._post_process_embedding(embed, seq_len)


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

        # self.model, self.alphabet = torch.hub.load(
        #     "facebookresearch/esm:main", model_name
        # )
        self.model.eval()
        # self.batch_converter = self.alphabet.get_batch_converter()
        self.repr_layer = repr_layer

    def clean(self, seq):
        if not re.match(r"^[ACDEFGHIKLMNPQRSTVWYX]+$", seq):
            print(f"Invalid sequence: {seq}")
            # convert unknown characters to X
            seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWYX]", "X", seq)
            print(f"Converted sequence: {seq}")
        return seq

    def prepare_sequences(self, sequences):
        # _, _, batch_tokens = self.batch_converter(
        #     [("", self.clean(seq)) for seq in sequences]
        # )
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
        # token_representations = results["hidden_states"][self.repr_layer]
        seq_lengths = input["attention_mask"].sum(1)

        for i, seq_len in enumerate(seq_lengths):
            yield token_representations[i, 1 : seq_len - 1]


class ESM3EmbeddingModel(BaseProteinEmbeddingModel):
    # largely taken from https://github.com/Dianzhuo-Wang/ esm3-structural-inputs/esm/
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
        self.tokenizer = self.model.tokenizers.sequence
        self.model.eval()
        self.use_norm_layer = use_norm_layer

        # Fix tokenizer if mask_token is None but exists in vocabulary
        self._fix_tokenizer_if_needed()

    def _fix_tokenizer_if_needed(self):
        """Fix the tokenizer if mask_token is None but exists in vocabulary (Singularity issue)."""
        if self.tokenizer.mask_token is None and "<mask>" in self.tokenizer.vocab:
            logger.info(
                "Fixing ESM3 tokenizer mask_token for Singularity compatibility"
            )
            self.tokenizer.mask_token = "<mask>"
            self.tokenizer.mask_token_id = self.tokenizer.vocab["<mask>"]

            # Also fix other tokenizers in the model if they exist
            for track in [
                "sequence",
                "structure",
                "secondary_structure",
                "sasa",
                "function",
                "residue_annotations",
            ]:
                if hasattr(self.model.tokenizers, track):
                    tokenizer = getattr(self.model.tokenizers, track)
                    if (
                        hasattr(tokenizer, "mask_token")
                        and tokenizer.mask_token is None
                        and hasattr(tokenizer, "vocab")
                        and "<mask>" in tokenizer.vocab
                    ):
                        tokenizer.mask_token = "<mask>"
                        tokenizer.mask_token_id = tokenizer.vocab["<mask>"]
                        logger.info(f"Fixed {track} tokenizer mask_token")

                    # Check that required tokens are set for empty tensor creation
                    if (
                        hasattr(tokenizer, "bos_token")
                        and tokenizer.bos_token is None
                        and hasattr(tokenizer, "vocab")
                        and "<cls>" in tokenizer.vocab
                    ):
                        tokenizer.bos_token = "<cls>"
                        tokenizer.bos_token_id = tokenizer.vocab["<cls>"]
                        logger.info(f"Fixed {track} tokenizer bos_token")

                    if (
                        hasattr(tokenizer, "eos_token")
                        and tokenizer.eos_token is None
                        and hasattr(tokenizer, "vocab")
                        and "<eos>" in tokenizer.vocab
                    ):
                        tokenizer.eos_token = "<eos>"
                        tokenizer.eos_token_id = tokenizer.vocab["<eos>"]
                        logger.info(f"Fixed {track} tokenizer eos_token")

    def _safe_encode(self, protein):
        """
        Custom encoding function that works around the None mask_token issue in Singularity.
        This is a safer alternative to model.encode() that works in both Docker and Singularity.
        """
        if protein.sequence is None:
            raise ValueError("sequence is required for encoding")

        # First check if we can use the standard encoding (if tokenizer is fixed)
        if self.tokenizer.mask_token is not None:
            try:
                return self.model.encode(protein)
            except Exception as e:
                logger.warning(
                    f"Standard encoding failed: {e}. Falling back to custom encoding."
                )

        # Fallback custom encoding approach
        sequence = protein.sequence
        # Replace underscore with X as a safe alternative
        sequence = sequence.replace("_", "X")

        # Use the tokenizer directly
        sequence_tokens = self.tokenizer.encode(sequence, add_special_tokens=True)
        sequence_tokens = torch.tensor(sequence_tokens, dtype=torch.int64)

        # Create the protein tensor with just the sequence
        protein_tensor = ESMProteinTensor(sequence=sequence_tokens)

        # Also handle structure information if provided in the input protein
        if protein.coordinates is not None:
            try:
                structure_encoder = self.model.get_structure_encoder()
                structure_tokenizer = self.model.tokenizers.structure

                coords, plddt, struct_tokens = encoding.tokenize_structure(
                    protein.coordinates,
                    structure_encoder,
                    structure_tokenizer=structure_tokenizer,
                    reference_sequence=protein.sequence,
                    add_special_tokens=True,
                )

                protein_tensor.coordinates = coords
                protein_tensor.structure = struct_tokens
            except Exception as e:
                logger.warning(f"Failed to encode structure: {e}")

        return protein_tensor

    def prepare_sequences(self, sequences, structure_path=None):
        """
        Prepare sequences and structures for ESM3 embedding.

        Args:
            sequences: List of protein sequences
            structures: Optional list of protein structures (paths to PDB files)

        Returns:
            List of ESMProteinTensor objects ready for embedding
        """
        # Convert sequences to ESMProtein objects and encode them using the safe encoder
        prots = [self._safe_encode(ESMProtein(sequence=seq)) for seq in sequences]

        # Process structures if provided
        if structure_path is not None:
            # Use single structure for all sequences
            # complex = ProteinComplex.from_pdb(structure_path)
            # # check if single chain or multiple chains
            # if len(complex.chains) > 1:
            #     # handle multiple chains, find chains with mutations
            #     for chain in complex.chains:
            #         pass
            #     protein = ESMProtein.from_complex(complex)
            # else:
            chain = ProteinChain.from_pdb(structure_path)
            protein = ESMProtein.from_protein_chain(chain)
            # select residues to encode in structure

            coords, _, struct_tokens = encoding.tokenize_structure(
                protein.coordinates,
                self.model.get_structure_encoder(),
                structure_tokenizer=self.model.tokenizers.structure,
                reference_sequence="",
                add_special_tokens=True,
            )
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

        Yields:
            Embeddings for each sequence (without special tokens)
        """
        # Make sure tokenizers are properly fixed before using them
        self._fix_tokenizer_if_needed()

        for protein_tensor in input:
            # Make a copy of the input
            protein_tensor = attr.evolve(protein_tensor)
            device = next(self.model.parameters()).device

            # Initialize default values for missing tracks
            try:
                # Create empty protein tensor with safer approach
                seq_len = len(protein_tensor) - 2  # minus special tokens

                # Custom implementation of empty tensor creation to avoid mask_token_id issues
                default_protein_tensor = self._create_empty_tensor(seq_len, device)

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
                        embeddings = self.model.transformer.norm(
                            forward_output.embeddings
                        )
                else:
                    embeddings = forward_output.embeddings

                # Check if embeddings is empty or has wrong shape
                if embeddings.shape[0] == 0:
                    logger.warning(
                        f"Got empty embeddings with shape {embeddings.shape}, attempting alternative approach"
                    )
                    # Try direct method - run model manually
                    sequence_tokens = protein_tensor.sequence.to(device).unsqueeze(
                        0
                    )  # Add batch dimension
                    attention_mask = torch.ones_like(sequence_tokens, dtype=torch.bool)

                    # Run model forward pass directly
                    outputs = self.model.transformer(
                        input_ids=sequence_tokens,
                        attention_mask=attention_mask,
                        return_dict=True,
                    )

                    # Get last hidden state
                    last_hidden = outputs.last_hidden_state

                    # Apply layer norm if requested
                    if self.use_norm_layer and hasattr(self.model.transformer, "norm"):
                        with (
                            torch.autocast(
                                device_type=device.type,
                                dtype=torch.bfloat16,
                                enabled=True,
                            )
                            if device.type == "cuda"
                            else contextlib.nullcontext()
                        ):
                            last_hidden = self.model.transformer.norm(last_hidden)

                    # Remove batch dimension and special tokens
                    embeddings = last_hidden[0, 1:-1]
                else:
                    # Remove special tokens (first and last)
                    if len(embeddings.shape) == 3 and embeddings.shape[0] > 0:
                        # If batched format: [batch_size, seq_len, hidden_dim]
                        embeddings = embeddings[0, 1:-1]
                    else:
                        # If unbatched: [seq_len, hidden_dim]
                        embeddings = embeddings[1:-1]

                # Return the embeddings
                yield embeddings.cpu()

            except Exception as e:
                logger.error(f"Error in forward pass: {e}")
                import traceback

                logger.error(traceback.format_exc())
                raise

    def _create_empty_tensor(self, length, device="cpu"):
        """
        Create an empty ESMProteinTensor with safe defaults for tokenizers.
        This method avoids using the encoding.get_default_*_tokens functions which require mask_token_id.
        """
        # Create minimal tensor with just sequence tokens
        sequence_tokenizer = self.model.tokenizers.sequence

        # Make sure the tokenizer has the required values
        if (
            sequence_tokenizer.mask_token_id is None
            and "<mask>" in sequence_tokenizer.vocab
        ):
            sequence_tokenizer.mask_token = "<mask>"
            sequence_tokenizer.mask_token_id = sequence_tokenizer.vocab["<mask>"]

        if (
            sequence_tokenizer.bos_token_id is None
            and "<cls>" in sequence_tokenizer.vocab
        ):
            sequence_tokenizer.bos_token = "<cls>"
            sequence_tokenizer.bos_token_id = sequence_tokenizer.vocab["<cls>"]

        if (
            sequence_tokenizer.eos_token_id is None
            and "<eos>" in sequence_tokenizer.vocab
        ):
            sequence_tokenizer.eos_token = "<eos>"
            sequence_tokenizer.eos_token_id = sequence_tokenizer.vocab["<eos>"]

        if sequence_tokenizer.mask_token_id is None:
            # If we can't fix the tokenizer, create a basic tensor with values from the vocabulary
            mask_id = sequence_tokenizer.vocab.get("<mask>", 32)
            bos_id = sequence_tokenizer.vocab.get("<cls>", 0)
            eos_id = sequence_tokenizer.vocab.get("<eos>", 2)

            # Create sequence tokens manually
            sequence_tokens = torch.full((length + 2,), mask_id, dtype=torch.int64)
            sequence_tokens[0] = bos_id
            sequence_tokens[-1] = eos_id
        else:
            # If the tokenizer is fixed, use the standard function
            sequence_tokens = torch.full(
                (length + 2,), sequence_tokenizer.mask_token_id, dtype=torch.int64
            )
            sequence_tokens[0] = sequence_tokenizer.bos_token_id
            sequence_tokens[-1] = sequence_tokenizer.eos_token_id

        # Return a minimal tensor with just the sequence
        return ESMProteinTensor(sequence=sequence_tokens.to(device))


class ESMCEmbeddingModel(BaseProteinEmbeddingModel):
    embedding_kind = EmbeddingType.PER_RESIDUE
    structure_aware = False

    def __init__(self, model_name: str):
        super().__init__()
        self.model: ESMC = ESMC.from_pretrained(model_name)
        self.model.eval()
        self.pad_idx = self.model.tokenizer.pad_token_id

    def prepare_sequences(self, sequences):
        # check if model is wrapped in DataParallel
        if isinstance(self.model, torch.nn.DataParallel):
            input_ids = self.model.module._tokenize(sequences)
        else:
            input_ids = self.model._tokenize(sequences)
        self.padding_mask = input_ids != self.pad_idx
        return input_ids

    @torch.no_grad()
    def forward(self, input):
        output: ESMCOutput = self.model(input)

        # Yield embeddings to maintain compatibility with other models
        for i in range(len(output.embeddings)):
            x = output.embeddings[i]  # get embedding for sequence i
            x = x[self.padding_mask[i]]  # remove padding tokens
            yield x[1:-1]  # remove start and end tokens


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

        for i, seq_len in enumerate(seq_lengths):
            yield token_representations[i, :seq_len]


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
