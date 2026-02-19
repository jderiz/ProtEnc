import torch
import torch.nn as nn
import protenc.utils as utils

from functools import cached_property
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

    def _get_data_loader(self, proteins: list[str], structures=None):
        assert isinstance(
            self.batch_size, int
        ), "batch size must be provided as integer at the moment"

        # Always pass structures to models - they can decide whether to use them
        def collate_fn(batch):
            return self.prepare_sequences([p for p in batch], structures)

        return self.dataloader(
            proteins,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            num_workers=self.preprocess_workers,
        )

    def prepare_sequences(self, proteins: list[str], structures=None):
        """Prepare protein sequences for encoding, optionally with structures."""
        # Always pass structures to models - they can decide whether to use them
        import inspect
        try:
            sig = inspect.signature(self.model.prepare_sequences)
            
            # Check which parameter name the model uses
            if 'structures' in sig.parameters:
                return self.model.prepare_sequences(proteins, structures=structures)
            elif 'structure_path' in sig.parameters:
                return self.model.prepare_sequences(proteins, structure_path=structures)
            else:
                # Models that don't accept structures parameter (shouldn't happen with base class)
                return self.model.prepare_sequences(proteins)
        except (AttributeError, TypeError):
            # Fallback if signature inspection fails
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
    ):
        """Process proteins in batches and yield embeddings."""
        batches = self._get_data_loader(proteins, structures)

        for batch in batches:
            # Move batch to device
            target_device = self._get_primary_device()
            if isinstance(batch, dict):
                # Move tensors to device
                batch = {
                    k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            elif isinstance(batch, list):
                # Handle list of tensors
                batch = [b.to(target_device) if hasattr(b, "to") else b for b in batch]
            else:
                batch = batch.to(target_device)

                # Handle model output (generators that yield embeddings)
            model_output = self._encode(batch)

            for embed in model_output:
                if average_sequence:
                    embed = embed.mean(0)
                yield utils.to_return_format(embed.cpu(), return_format)

    def encode(
        self,
        proteins: ProteinEncoderInput,
        structures=None,
        average_sequence: bool = True,  # mean over residue dimension
        return_format: ReturnFormat = "torch",
    ):
        """
        Encode proteins into embeddings.

        Args:
            proteins: List of protein sequences or dictionary with sequences as values
            structures: Optional path(s) to structure file(s) for structure-aware models
            average_sequence: Whether to average over the sequence dimension
            return_format: Format for the embeddings ("torch", "numpy", etc.)

        Yields:
            Embeddings for each protein in the requested format
        """
        if isinstance(proteins, dict):
            yield from zip(
                proteins.keys(),
                self.encode(
                    list(proteins.values()),
                    structures=structures,
                    average_sequence=average_sequence,
                    return_format=return_format,
                ),
            )
        elif isinstance(proteins, list):
            yield from self._encode_batches(
                proteins,
                structures=structures,
                average_sequence=average_sequence,
                return_format=return_format,
            )
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
