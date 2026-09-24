import warnings
from collections.abc import Mapping
from functools import cached_property

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import protenc.utils as utils
from protenc.models import BaseProteinEmbeddingModel, EmbeddingType, get_model
from protenc.types import BatchSize, ProteinEncoderInput, ReturnFormat


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
        dataloader: type[DataLoader] = DataLoader,
        data_parallel: bool = False,
        device_ids: list[int] | None = None,
        empty_cache_on_batch: bool = False,
    ):
        """
        Initialize the protein encoder.

        Args:
            model: The protein embedding model to use
            batch_size: Batch size for processing (default: 1)
            autocast: Whether to use automatic mixed precision
            preprocess_workers: Number of workers for data preprocessing. Not yet
                implemented; values greater than 0 are accepted but ignored.
            dataloader: DataLoader class to use
            data_parallel: Whether to use data parallel across all available GPUs
            device_ids: Optional explicit GPU ids for DataParallel (default: all visible GPUs)
            empty_cache_on_batch: If True, call ``torch.cuda.empty_cache()`` after each
                batch during encoding. Defaults to False because per-batch cache clearing
                hurts throughput; cache is still cleared on CUDA OOM recovery.
        """
        self.model: BaseProteinEmbeddingModel = model
        self.batch_size = 1 if batch_size is None else batch_size
        self._same_length_batch = getattr(model, "requires_same_length_batch", False)
        self.autocast = autocast  # Automatic mixed precision, saves memory and time at little accuracy cost
        self.preprocess_workers = preprocess_workers
        self.dataloader = dataloader
        self.data_parallel = data_parallel
        self.device_ids = device_ids
        self.empty_cache_on_batch = empty_cache_on_batch

        self._primary_device: torch.device | None = None

        # Replicate the model's compute core across GPUs if requested
        if self.data_parallel and torch.cuda.is_available():
            if self.device_ids is None:
                self.device_ids = list(range(torch.cuda.device_count()))
            try:
                self._primary_device = torch.device(f"cuda:{self.device_ids[0]}")
                self.model.to(self._primary_device)
                self.model.enable_data_parallel(self.device_ids)
            except Exception as e:
                warnings.warn(
                    f"Failed to initialize data parallel: {e}. Falling back to single GPU."
                )
                self.data_parallel = False
                self._primary_device = None

        # Validate data parallel setup
        if self.data_parallel:
            if not torch.cuda.is_available():
                warnings.warn("Data parallel requested but CUDA is not available")
            elif len(self.device_ids or []) < 2:
                warnings.warn("Data parallel requested but only one GPU is available")
            elif self.batch_size < len(self.device_ids):
                warnings.warn(
                    f"batch_size={self.batch_size} is smaller than the number of GPUs "
                    f"({len(self.device_ids)}); some GPUs will be idle."
                )

    @cached_property
    def device(self):
        """Get the device of the model, handling data parallel models."""
        if self.is_data_parallel and self._primary_device is not None:
            # For data parallel models, return the primary device (device_ids[0])
            return self._primary_device
        else:
            # For single device models, get device from parameters
            return next(iter(self.model.parameters())).device

    @property
    def is_data_parallel(self):
        """Check if the model is using data parallel."""
        return self.data_parallel and torch.cuda.is_available()

    def _get_primary_device(self):
        """Get the primary device for data parallel models."""
        return self.device

    def get_data_parallel_info(self):
        """Get information about the data parallel setup."""
        if not self.is_data_parallel:
            return {"enabled": False, "device_count": 1, "devices": [str(self.device)]}

        if self.device_ids is not None:
            devices = [f"cuda:{i}" for i in self.device_ids]
        else:
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

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
            elif len(self.device_ids or []) < 2:
                issues.append("Data parallel requested but only one GPU is available")

        return issues

    @property
    def chain_break_token(self) -> str:
        """
        Chain break token expected by the underlying embedding model.

        Defaults to empty string when the model does not define one.
        """
        return getattr(self.model, "chain_break_token", "")

    @property
    def repr_layer(self) -> int | None:
        """Representation layer used by the underlying embedding model."""
        return getattr(self.model, "repr_layer", None)

    @property
    def repr_layers(self) -> list[int] | None:
        """Layers extracted in the multi-layer path (None for single-layer)."""
        return getattr(self.model, "repr_layers", None)

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

    def _maybe_empty_cache(self, *, force: bool = False) -> None:
        """Clear CUDA cache when requested or after OOM recovery."""
        if (force or self.empty_cache_on_batch) and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run_encode_with_oom_recovery(self, encode_fn):
        """Run an encode callable, clearing CUDA cache and retrying once on OOM."""
        try:
            return encode_fn()
        except torch.cuda.OutOfMemoryError:
            self._maybe_empty_cache(force=True)
            return encode_fn()

    def _maybe_pool_embedding(
        self, embed: torch.Tensor, average_sequence: bool
    ) -> torch.Tensor:
        if not average_sequence:
            return embed
        if self.model.embedding_kind == EmbeddingType.PER_RESIDUE:
            if embed.ndim >= 2:
                return embed.mean(0)
        return embed

    def _encode(self, batch):
        """Process a batch through the model and return embeddings."""
        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device.type, enabled=self.autocast),
        ):
            # calls model.forward()
            return self.model(batch)

    def _encode_multi(self, batch, repr_layers: list[int]):
        """Process a batch through model.forward_multi for multiple layers."""
        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device.type, enabled=self.autocast),
        ):
            return self.model.forward_multi(batch, repr_layers)

    def _encode_batches(
        self,
        proteins: list[str],
        structures=None,
        average_sequence: bool = False,
        return_format: ReturnFormat = "torch",
        chain_list=None,
        structure_id=None,
        show_progress: bool = True,
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
            show_progress=show_progress,
        )

    def encode_sequences_batch(
        self,
        sequences: list[str],
        average_sequence: bool = False,
        return_format: ReturnFormat = "torch",
        show_progress: bool = False,
        structures=None,
        chain_list=None,
        structure_id=None,
    ):
        """
        Encode a single batch of raw protein sequences.

        Intended for bulk CLI pipelines that stream labeled batches from disk.
        Yields ``(local_index, embedding)`` tuples without nested progress bars.
        """
        yield from self._encode_batches(
            sequences,
            structures=structures,
            average_sequence=average_sequence,
            return_format=return_format,
            chain_list=chain_list,
            structure_id=structure_id,
            show_progress=show_progress,
        )

    def _batch_to_device(self, batch, target_device: torch.device):
        """Move batch tensors to device (mapping, list, or single tensor)."""
        if isinstance(batch, Mapping):  # includes transformers BatchEncoding
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
        show_progress: bool = True,
    ):
        """
        Prepare and encode one batch at a time: prepare_sequences -> forward -> yield.
        Yields (index, embed). Same flow for all models (ESM3, ESMC, etc.).
        """
        n = len(proteins)

        pbar_prepare = tqdm(
            total=n, desc="Preparing features", unit="seq", disable=not show_progress
        )
        pbar_embed = tqdm(
            total=n, desc="Embedding", unit="seq", disable=not show_progress
        )
        try:
            for batch_indices, batch_sequences in self._iter_batches(proteins):
                prepared_batch = self.prepare_sequences(
                    batch_sequences,
                    structures,
                    chain_list=chain_list,
                    structure_id=structure_id,
                )
                pbar_prepare.update(len(batch_indices))

                batch = self._batch_to_device(prepared_batch, target_device)
                model_output = self._run_encode_with_oom_recovery(
                    lambda b=batch: self._encode(b)
                )
                try:
                    for i, embed in enumerate(model_output):
                        embed = self._maybe_pool_embedding(embed, average_sequence)
                        out = utils.to_return_format(embed.cpu(), return_format)
                        del embed
                        yield (batch_indices[i], out)
                        pbar_embed.update(1)
                finally:
                    del model_output
                    del batch
                    del prepared_batch
                    self._maybe_empty_cache()
        finally:
            pbar_prepare.close()
            pbar_embed.close()

    def encode_multi_layer(
        self,
        proteins: list[str],
        repr_layers: list[int],
        structures=None,
        average_sequence: bool = False,
        return_format: ReturnFormat = "torch",
        chain_list=None,
        structure_id=None,
        show_progress: bool = True,
    ):
        """
        Encode proteins for multiple layers in a single forward pass per batch.

        Yields (index, layer, embed) so the consumer can route each layer to its
        own writer. Prepares and encodes one batch at a time without retaining
        all prepared tensors in memory.
        """
        target_device = self._get_primary_device()
        n = len(proteins)

        pbar_prepare = tqdm(
            total=n, desc="Preparing features", unit="seq", disable=not show_progress
        )
        pbar_embed = tqdm(
            total=n * len(repr_layers),
            desc="Embedding",
            unit="emb",
            disable=not show_progress,
        )
        try:
            for batch_indices, batch_sequences in self._iter_batches(proteins):
                prepared_batch = self.prepare_sequences(
                    batch_sequences,
                    structures,
                    chain_list=chain_list,
                    structure_id=structure_id,
                )
                pbar_prepare.update(len(batch_indices))

                batch = self._batch_to_device(prepared_batch, target_device)
                model_output = self._run_encode_with_oom_recovery(
                    lambda b=batch: self._encode_multi(b, repr_layers)
                )
                try:
                    for i, layer, embed in model_output:
                        embed = self._maybe_pool_embedding(embed, average_sequence)
                        out = utils.to_return_format(embed.cpu(), return_format)
                        del embed
                        yield (batch_indices[i], layer, out)
                        pbar_embed.update(1)
                finally:
                    del model_output
                    del batch
                    del prepared_batch
                    self._maybe_empty_cache()
        finally:
            pbar_prepare.close()
            pbar_embed.close()

    def encode(
        self,
        proteins: ProteinEncoderInput,
        structures=None,
        average_sequence: bool = True,  # mean over residue dimension
        return_format: ReturnFormat = "torch",
        chain_list=None,
        structure_id=None,
        show_progress: bool = True,
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
                show_progress=show_progress,
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
                show_progress=show_progress,
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

        batch = self._batch_to_device(batch, self._get_primary_device())

        # Get embeddings from generator
        model_output = self._encode(batch)
        embeds = list(model_output)

        # For batched output, stack the embeddings
        if len(embeds) > 1:
            stacked_embeds = torch.stack(embeds)
            if (
                average_sequence
                and self.model.embedding_kind == EmbeddingType.PER_RESIDUE
            ):
                stacked_embeds = stacked_embeds.mean(1)
            return utils.to_return_format(stacked_embeds.cpu(), return_format)
        else:
            embed = self._maybe_pool_embedding(embeds[0], average_sequence)
            return utils.to_return_format(embed.cpu(), return_format)

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)


def get_encoder(
    model_name,
    device=None,
    repr_layer=None,
    repr_layers=None,
    data_parallel=False,
    device_ids=None,
    **kwargs,
):
    """
    Create a ProteinEncoder instance with the specified model.

    Args:
        model_name: Name of the model to load
        device: Device to place the model on
        repr_layer: Optional 1-indexed transformer layer for representations
        repr_layers: Optional list of 1-indexed layers for multi-layer extraction
        data_parallel: Whether to use data parallel across all available GPUs
        device_ids: Optional explicit GPU ids for DataParallel
        **kwargs: Additional arguments to pass to ProteinEncoder

    Returns:
        ProteinEncoder instance
    """
    model = get_model(model_name, repr_layer=repr_layer, repr_layers=repr_layers)

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

    return ProteinEncoder(
        model, data_parallel=data_parallel, device_ids=device_ids, **kwargs
    )
