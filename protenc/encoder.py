import torch
import protenc.utils as utils

from functools import cached_property
from protenc.types import BatchSize, ProteinEncoderInput, ReturnFormat
from torch.utils.data import DataLoader
from protenc.models import BaseProteinEmbeddingModel, get_model


class ProteinEncoder:
    def __init__(
        self,
        model: BaseProteinEmbeddingModel,
        batch_size: BatchSize = None,
        autocast: bool = False,
        preprocess_workers: int = 0,
        dataloader: DataLoader = DataLoader,
    ):
        self.model = model
        self.batch_size = 1 if batch_size is None else batch_size
        self.autocast = autocast  # Automatic mixed precision, saves memory and time at little accuracy cost
        self.preprocess_workers = preprocess_workers
        self.dataloader = dataloader

    @cached_property
    def device(self):
        return next(iter(self.model.parameters())).device

    def _get_data_loader(self, proteins: list[str], structures=None):
        assert isinstance(
            self.batch_size, int
        ), "batch size must be provided as integer at the moment"

        # Create a collate function that passes structures if they are provided
        if (
            structures is not None
            and hasattr(self.model, "structure_aware")
            and self.model.structure_aware
        ):

            def collate_with_structures(batch):
                return self.prepare_sequences([p for p in batch], structures)

            collate_fn = collate_with_structures
        else:

            def collate_without_structures(batch):
                return self.prepare_sequences(batch)

            collate_fn = collate_without_structures

        return self.dataloader(
            proteins,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            num_workers=self.preprocess_workers,
        )

    def prepare_sequences(self, proteins: list[str], structure_path=None):
        """Prepare protein sequences for encoding, optionally with structures."""
        if structure_path is not None and self.model.structure_aware:
            return self.model.prepare_sequences(proteins, structure_path)
        else:
            return self.model.prepare_sequences(proteins)

    def _encode(self, batch):
        """Process a batch through the model and return embeddings."""
        with torch.inference_mode(), torch.autocast("cuda", enabled=self.autocast):
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
            if isinstance(batch, dict):
                # Move tensors to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            elif isinstance(batch, list):
                # Handle list of tensors 
                batch = [b.to(self.device) if hasattr(b, "to") else b for b in batch]
            else:
                batch = batch.to(self.device)

            for embed in self._encode(batch):
                if average_sequence:
                    embed = embed.mean(0)

                yield utils.to_return_format(embed.cpu(), return_format)

    def encode(
        self,
        proteins: ProteinEncoderInput,
        structures=None,
        average_sequence: bool = False,  # actually average over tokens -> sequence embedding
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
        if isinstance(batch, dict):
            # Move tensors to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        elif isinstance(batch, list):
            # Handle list of tensors (for ESM3)
            batch = [b.to(self.device) if hasattr(b, "to") else b for b in batch]
        else:
            batch = batch.to(self.device)

        # Get embeddings
        embeds = list(self._encode(batch))

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


def get_encoder(model_name, device=None, **kwargs):
    model = get_model(model_name)

    if device is not None:
        model = model.to(device)

    return ProteinEncoder(model, **kwargs)
