from dataclasses import dataclass
import os
import torch
import torch.nn as nn
import attr
import contextlib
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
from esm.models.esmc import ESMC, ESMCOutput
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, ESMProteinTensor, ProteinComplex
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils import encoding
from esm.utils.sampling import _BatchedESMProteinTensor
from esm.utils.generation import _batch_forward

try:
    from .mpnn import ProteinMPNN, parse_PDB, tied_featurize, gather_nodes
except ImportError:
    from mpnn import ProteinMPNN, parse_PDB, tied_featurize, gather_nodes

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

    def prepare_sequences(self, sequences, structures=None):
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

    def prepare_sequences(self, sequences, structures=None):
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
    """ESM3 embedding model that can handle sequences and structures."""
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
        self.model_name = model_name

    def _get_model(self):
        """Get the underlying model, handling DataParallel wrapping."""
        return self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

    def _safe_encode(self, protein: ESMProtein) -> ESMProteinTensor:
        """
        Encode an ESMProtein to ESMProteinTensor using the model's encode method.
        """
        if protein.sequence is None:
            raise ValueError("sequence is required for encoding")

        try:
            return self._get_model().encode(protein)
        except Exception as e:
            logger.warning(f"Standard encoding failed: {e}. Using fallback encoding.")
            
            # Fallback: create tensor manually
            sequence_tokenizer = self._get_model().tokenizers.sequence
            sequence_tokens = sequence_tokenizer.encode(protein.sequence, add_special_tokens=True)
            sequence_tokens = torch.tensor(sequence_tokens, dtype=torch.int64)
            
            protein_tensor = ESMProteinTensor(sequence=sequence_tokens)
            
            # Handle structure if provided
            if protein.coordinates is not None:
                try:
                    coords, plddt, struct_tokens = encoding.tokenize_structure(
                        protein.coordinates,
                        self._get_model().get_structure_encoder(),
                        structure_tokenizer=self._get_model().tokenizers.structure,
                        reference_sequence=protein.sequence,
                        add_special_tokens=True,
                    )
                    protein_tensor.coordinates = coords
                    protein_tensor.structure = struct_tokens
                except Exception as e:
                    logger.warning(f"Failed to encode structure: {e}")
                    
            return protein_tensor

    def prepare_sequences(
        self,
        sequences: List[str],
        structures=None,
        chain_list: Optional[List[str]] = None,
        structure_id: Optional[str] = None,
    ):
        """
        Prepare sequences and structures for ESM3 embedding.

        Encoding and structure handling mirror haipr/models/esm3.py:
        - Uses ProteinComplex (full or filtered by chain_list with optional structure_id).
        - When structure is present: tokenizes structure once with reference_sequence
          from the first protein, shares structure tokens and coordinates across all
          proteins, and builds one ESMProteinTensor per sequence with
          encoding.tokenize_sequence / encoding.tokenize_structure.
        - When no structure: encodes each sequence via model.encode (or _safe_encode).

        Args:
            sequences: List of protein sequences (may use "|" for multi-chain).
            structures: Optional path to structure file for structure-aware encoding
                (string or single-element list).
            chain_list: Optional list of chain IDs to load (None = full complex).
            structure_id: Optional structure id passed to ProteinChain.from_pdb when
                loading by chain_list (mirrors id in haipr create_filtered_protein_complex).

        Returns:
            List of ESMProteinTensor objects ready for embedding.
        """
        structure_path = (
            structures
            if isinstance(structures, str)
            else (structures[0] if structures else None)
        )

        if structure_path is None:
            proteins = [ESMProtein(sequence=seq) for seq in sequences]
            return [self._safe_encode(protein) for protein in proteins]

        try:
            if chain_list is not None:
                pc = _create_filtered_protein_complex(
                    structure_path, chain_list, id=structure_id
                )
            else:
                pc = ProteinComplex.from_pdb(structure_path)

            protein_with_structure = ESMProtein.from_protein_complex(pc)
            shared_coordinates = protein_with_structure.coordinates

            proteins = [
                ESMProtein(sequence=seq, coordinates=shared_coordinates)
                for seq in sequences
            ]

            first = proteins[0]
            if first.coordinates is not None:
                shared_coords, _, shared_structure_tokens = encoding.tokenize_structure(
                    first.coordinates,
                    self._get_model().get_structure_encoder(),
                    structure_tokenizer=self._get_model().tokenizers.structure,
                    reference_sequence=first.sequence or "",
                    add_special_tokens=True,
                )
                protein_tensors = []
                for p in proteins:
                    sequence_tokens = encoding.tokenize_sequence(
                        p.sequence or "",
                        self._get_model().tokenizers.sequence,
                        add_special_tokens=True,
                    )
                    if not isinstance(sequence_tokens, torch.Tensor):
                        sequence_tokens = torch.tensor(
                            sequence_tokens, dtype=torch.long
                        )
                    pt = ESMProteinTensor(
                        sequence=sequence_tokens,
                        structure=shared_structure_tokens,
                        secondary_structure=None,
                        sasa=None,
                        function=None,
                        residue_annotations=None,
                        coordinates=shared_coords,
                    )
                    protein_tensors.append(pt)
                return protein_tensors

        except Exception as e:
            logger.warning(f"Failed to load structure from {structure_path}: {e}")

        proteins = [ESMProtein(sequence=seq) for seq in sequences]
        return [self._safe_encode(protein) for protein in proteins]

    @torch.no_grad()
    def forward(self, input):
        """
        Generate embeddings for the input sequences.

        Args:
            input: List of ESMProteinTensor objects

        Yields:
            Embeddings for each sequence (without special tokens)
        """
        device = next(self.model.parameters()).device
        model = self._get_model()

        for protein_tensor in input:
            try:
                # Move to device and prepare input
                protein_tensor = attr.evolve(protein_tensor)
                
                # Prepare inputs for forward pass
                inputs = {}
                
                # Always include sequence
                if protein_tensor.sequence is not None:
                    inputs["sequence_tokens"] = protein_tensor.sequence.to(device).unsqueeze(0)
                
                # Include structure if available
                if hasattr(protein_tensor, 'structure') and protein_tensor.structure is not None:
                    inputs["structure_tokens"] = protein_tensor.structure.to(device).unsqueeze(0)
                
                # Include coordinates if available
                if hasattr(protein_tensor, 'coordinates') and protein_tensor.coordinates is not None:
                    inputs["structure_coords"] = protein_tensor.coordinates.to(device).unsqueeze(0)

                # Run forward pass using the model directly
                # internal forward handles normalization and other steps
                output = model(**inputs)
                embeddings = output.embeddings

                # Remove batch dimension and special tokens
                if len(embeddings.shape) == 3:
                    # [batch_size, seq_len, hidden_dim] -> [seq_len, hidden_dim]
                    embeddings = embeddings[0, 1:-1]
                else:
                    # [seq_len, hidden_dim]
                    embeddings = embeddings[1:-1]

                yield embeddings.cpu()

            except Exception as e:
                logger.error(f"Error in forward pass: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise


class ESMCEmbeddingModel(BaseProteinEmbeddingModel):
    embedding_kind = EmbeddingType.PER_RESIDUE
    structure_aware = False

    def __init__(self, model_name: str):
        super().__init__()
        self.model: ESMC = ESMC.from_pretrained(model_name)
        self.model.eval()
        self.pad_idx = self.model.tokenizer.pad_token_id

    def prepare_sequences(self, sequences, structures=None):
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

    def prepare_sequences(self, sequences, structures=None):
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


class ProteinMPNNEmbeddingModel(BaseProteinEmbeddingModel):
    """ProteinMPNN embedding model that requires structure information."""
    embedding_kind = EmbeddingType.PER_RESIDUE
    structure_aware = True

    def __init__(self, model_name: str, ca_only: bool = False, use_structure: bool = True):
        """
        Initialize the ProteinMPNN embedding model.
        
        Args:
            model_name: Path to the ProteinMPNN checkpoint file (.pt)
            ca_only: Whether to use CA-only model (default: False)
        """
        super().__init__()
        print(f"Initializing ProteinMPNN embedding model with model name: {model_name}")
        # get torch model cache location (local model cache)and prepend it to the model.
        # TODO: add automatic download if model not found in cache.
        cache_dir = torch.hub.get_dir()
        if os.path.exists(os.path.join(cache_dir,"ProteinMPNN", "weights", model_name)):
            model_name = os.path.join(cache_dir,"ProteinMPNN", "weights", model_name)
        else:
            # this_file_path/weights/model_name.pt
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
        pdb_dict_list = parse_PDB(structures, ca_only=self.ca_only)
        logger.debug(f"pdb_dict_list: {len(pdb_dict_list)}")
        if not pdb_dict_list:
            raise ValueError(f"Failed to parse PDB file: {structures}")
        
        pdb_dict = pdb_dict_list[0]
        pdb_seq_len = len(pdb_dict.get('seq', ''))
        
        # Find all chain letters in the PDB dict
        chain_keys = [key for key in pdb_dict.keys() if key.startswith('seq_chain_')]
        chain_letters = [key.replace('seq_chain_', '') for key in chain_keys]
        
        # If sequences provided, check if all have same length as PDB
        if sequences:
            if not all(len(seq) == pdb_seq_len for seq in sequences):
                raise ValueError(f"All sequences ({len(sequences[0])}) must have the same length as PDB structure ({pdb_seq_len})")
            # Create batch with same structure but different sequences
            batch = []
            for seq in sequences:
                seq_dict = pdb_dict.copy()
                seq_dict['seq'] = seq
                # Update all seq_chain_{letter} entries with the new sequence
                # For multi-chain proteins, we need to split the sequence appropriately
                # For now, assume single chain or concatenated sequence matches the PDB structure
                if len(chain_letters) == 1:
                    # Single chain: replace the entire chain sequence
                    seq_dict[f'seq_chain_{chain_letters[0]}'] = seq
                else:
                    # Multi-chain: need to split sequence by chain lengths
                    # This is a simplified approach - assumes chains are concatenated in order
                    start_idx = 0
                    for letter in chain_letters:
                        chain_key = f'seq_chain_{letter}'
                        if chain_key in pdb_dict:
                            chain_len = len(pdb_dict[chain_key])
                            seq_dict[chain_key] = seq[start_idx:start_idx + chain_len]
                            start_idx += chain_len
                batch.append(seq_dict)
        else:
            batch = [pdb_dict]

        
        # Featurize using mpnn function
        featurized = tied_featurize(
            batch, device, chain_dict=None, fixed_position_dict=None,
            omit_AA_dict=None, tied_positions_dict=None, pssm_dict=None,
            bias_by_res_dict=None, ca_only=self.ca_only
        )
        
        return {
            'X': featurized[0],
            'S': featurized[1],
            'mask': featurized[2],
            'chain_M': featurized[4],
            'residue_idx': featurized[12],
            'chain_encoding_all': featurized[5],
            'lengths': featurized[3],
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
        # Follow same logic for feature preparation as in ProteinMPNN in mpnn.py
        device = next(self.model.parameters()).device
        X = input['X'].to(device)
        S = input['S'].to(device)
        mask = input['mask'].to(device)
        residue_idx = input['residue_idx'].to(device)
        chain_encoding_all = input['chain_encoding_all'].to(device)
        lengths = input['lengths']

        # --- Feature Preparation (matches ProteinMPNN) ---
        # Get edge features and indices with ProteinFeatures
        E, E_idx = self.model.features(X, mask, residue_idx, chain_encoding_all)
        
        # Sequence and edge projections
        h_S = self.model.W_s(S)
        h_V = h_S.clone()  # Initialize h_V with sequence embeddings
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

    @classmethod
    def from_model_cls(cls, *, model_cls, model_kwargs, **kwargs):
        def init_fn(**init_kwargs):
            return model_cls(**{**model_kwargs, **init_kwargs})

        return cls(init_fn=init_fn, **kwargs)


model_descriptions = [
    # CARP family (https://github.com/microsoft/protein-sequence-models)
    ModelCard.from_model_cls(
        name="carp",
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
    # ProteinMPNN
    ModelCard.from_model_cls(
        name="mpnn",
        family="ProteinMPNN",
        embed_dim=128,
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
    }


def get_model(model_name, **kwargs):
    model = model_dict[model_name].init_fn(**kwargs)

    return model
