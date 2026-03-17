import torch
import torch.nn as nn
import protenc.utils as utils

from functools import cached_property
from tqdm import tqdm
from protenc.types import BatchSize, ProteinEncoderInput, ReturnFormat
from torch.utils.data import DataLoader
from protenc.models import BaseProteinEmbeddingModel, get_model
from esm.models.esmc import ESMC


class ProteinEncoder:
    """
    A protein encoder that can process protein sequences using various embedding models.

    Supports data parallel processing across multiple GPUs for improved performance.
    """

    def __init__(
        self,
        model: BaseProteinEmbeddingModel,
        batch_size: BatchSize = None,
        autocast: bool = False,
        preprocess_workers: int = 0,
        dataloader: DataLoader = DataLoader,
        data_parallel: bool = False,
    ):
        """
        Initialize the protein encoder.

        Args:
            model: The protein embedding model to use
            batch_size: Batch size for processing (default: 1)
            autocast: Whether to use automatic mixed precision
            preprocess_workers: Number of workers for data preprocessing
            dataloader: DataLoader class to use
            data_parallel: Whether to use data parallel across all available GPUs
        """
        self.model = model
        self.batch_size = 1 if batch_size is None else batch_size
        self._same_length_batch = getattr(model, "requires_same_length_batch", False)
        self.autocast = autocast  # Automatic mixed precision, saves memory and time at little accuracy cost
        self.preprocess_workers = preprocess_workers
        self.dataloader = dataloader
        self.data_parallel = data_parallel

        # Apply data parallel if requested and CUDA is available
        if self.data_parallel and torch.cuda.is_available():
            # Check if this is an ESMC model - DataParallel doesn't work well with ESMC
            is_esmc_model = False
            if hasattr(self.model, "model"):
                if hasattr(self.model.model, "model") and isinstance(
                    self.model.model.model, ESMC
                ):
                    is_esmc_model = True
                elif isinstance(self.model.model, ESMC):
                    is_esmc_model = True
            elif hasattr(self.model, "model") and isinstance(self.model.model, ESMC):
                is_esmc_model = True

            if is_esmc_model:
                import warnings

                warnings.warn(
                    "DataParallel is not supported for ESMC models due to ESMCOutput compatibility issues. "
                    "Falling back to single GPU for ESMC models."
                )
                self.data_parallel = False
            else:
                # Apply DataParallel for other models
                try:
                    if hasattr(self.model, "model"):
                        # Some models wrap the actual model in a .model attribute
                        self.model.model = nn.DataParallel(self.model.model)
                    else:
                        # Direct model wrapping
                        self.model = nn.DataParallel(self.model)
                except Exception as e:
                    import warnings

                    warnings.warn(
                        f"Failed to initialize data parallel: {e}. Falling back to single GPU."
                    )
                    self.data_parallel = False

        # Validate data parallel setup
        if self.data_parallel:
            import warnings

            if not torch.cuda.is_available():
                warnings.warn("Data parallel requested but CUDA is not available")
            elif torch.cuda.device_count() < 2:
                warnings.warn("Data parallel requested but only one GPU is available")

    @cached_property
    def device(self):
        """Get the device of the model, handling data parallel models."""
        if self.data_parallel and torch.cuda.is_available():
            # For data parallel models, return the primary device (cuda:0)
            return torch.device("cuda:0")
        else:
            # For single device models, get device from parameters
            return next(iter(self.model.parameters())).device

    @property
    def is_data_parallel(self):
        """Check if the model is using data parallel."""
        return self.data_parallel and torch.cuda.is_available()

    def _get_primary_device(self):
        """Get the primary device for data parallel models."""
        if self.is_data_parallel:
            return torch.device("cuda:0")
        return self.device

    def get_data_parallel_info(self):
        """Get information about the data parallel setup."""
        if not self.is_data_parallel:
            return {"enabled": False, "device_count": 1, "devices": [str(self.device)]}

        device_count = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(device_count)]

        return {
            "enabled": True,
            "device_count": len(devices),
            "devices": devices,
            "primary_device": str(self._get_primary_device()),
        }

    def validate_data_parallel_setup(self):
        """Validate the data parallel setup and return any issues."""
        issues = []

        if self.data_parallel:
            if not torch.cuda.is_available():
                issues.append("Data parallel requested but CUDA is not available")
            elif torch.cuda.device_count() < 2:
                issues.append("Data parallel requested but only one GPU is available")

        return issues

    @property
    def chain_break_token(self) -> str:
        """
        Chain break token expected by the underlying embedding model.

        Defaults to empty string when the model does not define one.
        """
        model = self.model
        # When DataParallel is applied, the wrapped module may live under .module
        if isinstance(model, nn.DataParallel):
            model = model.module
        return getattr(model, "chain_break_token", "")

    def _iter_batches(self, proteins: list[str]):
        """Iterate (batch_indices, batch_sequences). Same-length models: group by length then chunk; else consecutive chunks."""
        assert isinstance(self.batch_size, int), "batch size must be an integer"
        n = len(proteins)
        if self._same_length_batch:
            from collections import defaultdict
            by_len = defaultdict(list)
            for i, p in enumerate(proteins):
                by_len[len(p)].append(i)
            for _length in sorted(by_len.keys()):
                indices_L = by_len[_length]
                for start in range(0, len(indices_L), self.batch_size):
                    chunk = indices_L[start : start + self.batch_size]
                    yield chunk, [proteins[j] for j in chunk]
        else:
            for start in range(0, n, self.batch_size):
                chunk = list(range(start, min(start + self.batch_size, n)))
                yield chunk, [proteins[j] for j in chunk]

    def prepare_sequences(
        self,
        proteins: list[str],
        structures=None,
        chain_list=None,
        structure_id=None,
    ):
        """Prepare protein sequences for encoding, optionally with structures."""
        import inspect
        try:
            sig = inspect.signature(self.model.prepare_sequences)
            kwargs = {}
            if "structures" in sig.parameters:
                kwargs["structures"] = structures
            elif "structure_path" in sig.parameters:
                kwargs["structure_path"] = structures
            if chain_list is not None and "chain_list" in sig.parameters:
                kwargs["chain_list"] = chain_list
            if structure_id is not None and "structure_id" in sig.parameters:
                kwargs["structure_id"] = structure_id
            if kwargs:
                return self.model.prepare_sequences(proteins, **kwargs)
            return self.model.prepare_sequences(proteins)
        except (AttributeError, TypeError):
            return self.model.prepare_sequences(proteins, structures=structures)

    def _encode(self, batch):
        """Process a batch through the model and return embeddings."""
        with torch.inference_mode(), torch.amp.autocast(
            device_type="cuda", enabled=self.autocast
        ):
            # calls model.forward()
            return self.model(batch)

    def _encode_batches(
        self,
        proteins: list[str],
        structures=None,
        average_sequence: bool = False,
        return_format: ReturnFormat = "torch",
        chain_list=None,
        structure_id=None,
    ):
        """
        Flow for all models: 1) prepare_sequences (tokenization), 2) batch-wise forward.
        Yields (index, embed) so the consumer can write to disk by index (streamed).
        """
        target_device = self._get_primary_device()
        yield from self._encode_two_phase(
            proteins,
            structures,
            average_sequence,
            return_format,
            target_device,
            chain_list=chain_list,
            structure_id=structure_id,
        )

    def _batch_to_device(self, batch, target_device: torch.device):
        """Move batch tensors to device (dict, list, or single tensor)."""
        if isinstance(batch, dict):
            return {
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        if isinstance(batch, list):
            return [b.to(target_device) if hasattr(b, "to") else b for b in batch]
        return batch.to(target_device)

    def _encode_two_phase(
        self,
        proteins: list[str],
        structures,
        average_sequence: bool,
        return_format: ReturnFormat,
        target_device: torch.device,
        chain_list=None,
        structure_id=None,
    ):
        """
        Phase 1: prepare_sequences for all sequences (batched via _iter_batches).
        Phase 2: batch-wise forward. Yields (index, embed).
        Same flow for all models (ESM3, ESMC, etc.).
        """
        n = len(proteins)
        stored_batches = []

        pbar_prepare = tqdm(total=n, desc="Preparing features", unit="seq")
        try:
            for batch_indices, batch_sequences in self._iter_batches(proteins):
                prepared_batch = self.prepare_sequences(
                    batch_sequences,
                    structures,
                    chain_list=chain_list,
                    structure_id=structure_id,
                )
                stored_batches.append((batch_indices, prepared_batch))
                pbar_prepare.update(len(batch_indices))
        finally:
            pbar_prepare.close()

        pbar_embed = tqdm(total=n, desc="Embedding", unit="seq")
        try:
            for batch_indices, prepared_batch in stored_batches:
                batch = self._batch_to_device(prepared_batch, target_device)
                model_output = self._encode(batch)
                try:
                    for i, embed in enumerate(model_output):
                        if average_sequence:
                            embed = embed.mean(0)
                        out = utils.to_return_format(embed.cpu(), return_format)
                        del embed
                        yield (batch_indices[i], out)
                        pbar_embed.update(1)
                finally:
                    del model_output
                    del batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            pbar_embed.close()

    def encode(
        self,
        proteins: ProteinEncoderInput,
        structures=None,
        average_sequence: bool = True,  # mean over residue dimension
        return_format: ReturnFormat = "torch",
        chain_list=None,
        structure_id=None,
    ):
        """
        Encode proteins into embeddings.

        Args:
            proteins: List of protein sequences or dictionary with sequences as values
            structures: Optional path(s) to structure file(s) for structure-aware models
            average_sequence: Whether to average over the sequence dimension
            return_format: Format for the embeddings ("torch", "numpy", etc.)
            chain_list: Optional list of chain IDs in order (for structure-aware models)
            structure_id: Optional structure id for ESM loading by chain_list

        Yields:
            Embeddings for each protein in the requested format
        """
        if isinstance(proteins, dict):
            keys = list(proteins.keys())
            seqs = list(proteins.values())
            gen = self._encode_batches(
                seqs,
                structures=structures,
                average_sequence=average_sequence,
                return_format=return_format,
                chain_list=chain_list,
                structure_id=structure_id,
            )
            for _idx, emb in gen:
                yield keys[_idx], emb
        elif isinstance(proteins, list):
            gen = self._encode_batches(
                proteins,
                structures=structures,
                average_sequence=average_sequence,
                return_format=return_format,
                chain_list=chain_list,
                structure_id=structure_id,
            )
            for item in gen:
                yield item  # (index, embed) for streamed write-by-index
        else:
            raise TypeError(
                "Expected list of proteins sequences or dictionary with protein "
                f"sequences as values but found {type(proteins)}"
            )

    def encode_batch(
        self,
        proteins: list[str],
        structures=None,
        average_sequence: bool = False,
        return_format: ReturnFormat = "torch",
    ):
        """
        Encode a batch of proteins at once.

        When invoked from HAIPR, ``proteins`` should be a list of
        sequences in HAIPR embedding format (see
        ``haipr.data.HAIPRData.get_sequences_for_embedding``), and, when
        ``structures`` is provided, each sequence length is expected to
        match the residue count of the corresponding structure for the
        loaded chains.

        Args:
            proteins: List of protein sequences
            structures: Optional path(s) to structure file(s) for structure-aware models
            average_sequence: Whether to average over the sequence dimension
            return_format: Format for the embeddings

        Returns:
            Embeddings for the batch in the requested format
        """
        batch = self.prepare_sequences(proteins, structures)

        # Move batch to device
        target_device = self._get_primary_device()
        if isinstance(batch, dict):
            # Move tensors to device
            batch = {
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        elif isinstance(batch, list):
            # Handle list of tensors (for ESM3)
            batch = [b.to(target_device) if hasattr(b, "to") else b for b in batch]
        else:
            batch = batch.to(target_device)

        # Get embeddings from generator
        model_output = self._encode(batch)
        embeds = list(model_output)

        # For batched output, stack the embeddings
        if len(embeds) > 1:
            stacked_embeds = torch.stack(embeds)
            if average_sequence:
                stacked_embeds = stacked_embeds.mean(1)
            return utils.to_return_format(stacked_embeds.cpu(), return_format)
        else:
            embed = embeds[0]
            if average_sequence:
                embed = embed.mean(0)
            return utils.to_return_format(embed.cpu(), return_format)

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)


def get_encoder(model_name, device=None, data_parallel=False, **kwargs):
    """
    Create a ProteinEncoder instance with the specified model.

    Args:
        model_name: Name of the model to load
        device: Device to place the model on
        data_parallel: Whether to use data parallel across all available GPUs
        **kwargs: Additional arguments to pass to ProteinEncoder

    Returns:
        ProteinEncoder instance
    """
    model = get_model(model_name)

    # Validate and handle device parameter
    if device is not None:
        # Handle string device specifications
        if isinstance(device, str):
            # Convert common device strings to proper format
            if device.lower() in ["none", "null", ""]:
                device = None
            elif device.lower() == "cuda":
                device = "cuda:0"  # Default to first GPU
            elif device.lower().startswith("cuda"):
                # Ensure proper cuda device format
                if ":" not in device:
                    device = f"{device}:0"

        if device is not None:
            try:
                model = model.to(device)
            except Exception as e:
                raise ValueError(f"Invalid device specification '{device}': {e}")

    return ProteinEncoder(model, data_parallel=data_parallel, **kwargs)
