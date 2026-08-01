"""MCP server exposing ProtEnc protein embedding tools to AI agents."""

from __future__ import annotations

import base64
import logging
import os
import sys
from typing import Any

import numpy as np
from mcp.server.fastmcp import FastMCP

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "protenc",
    instructions=(
        "Generate protein sequence embeddings with ProtEnc. "
        "Use protenc_list_models to discover models, protenc_get_model_info for "
        "layer and dimension details, and protenc_embed_sequences to compute embeddings."
    ),
)

_ENCODER_CACHE: dict[tuple[str, str, int | None], Any] = {}


def _default_device() -> str:
    configured = os.environ.get("PROTENC_DEVICE")
    if configured:
        return configured

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _default_model() -> str:
    return os.environ.get("PROTENC_DEFAULT_MODEL", "esm2_t30")


def _get_encoder(model_name: str, device: str, repr_layer: int | None):
    cache_key = (model_name, device, repr_layer)
    if cache_key not in _ENCODER_CACHE:
        import protenc

        logger.info(
            "Loading ProtEnc model '%s' on device '%s' (repr_layer=%s)",
            model_name,
            device,
            repr_layer,
        )
        _ENCODER_CACHE[cache_key] = protenc.get_encoder(
            model_name,
            device=device,
            repr_layer=repr_layer,
        )
    return _ENCODER_CACHE[cache_key]


def _serialize_embedding(embedding: np.ndarray) -> dict[str, Any]:
    array = np.asarray(embedding, dtype=np.float32)
    payload: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": "float32",
    }

    if array.size <= 16_384:
        payload["embedding"] = array.tolist()
    else:
        payload["embedding_b64"] = base64.b64encode(array.tobytes()).decode("ascii")
        payload["note"] = (
            "Embedding returned as base64-encoded float32 bytes because the vector "
            "is large. Decode with numpy.frombuffer(base64.b64decode(...), dtype=np.float32)"
            f".reshape{tuple(array.shape)}."
        )

    return payload


@mcp.tool()
def protenc_list_models(family: str | None = None) -> dict[str, Any]:
    """List protein embedding models available in ProtEnc.

    Args:
        family: Optional model family filter (for example "ESM" or "ProtTrans").
    """
    import protenc

    models = protenc.list_models(family=family)
    return {"models": models, "count": len(models), "family_filter": family}


@mcp.tool()
def protenc_get_model_info(model_name: str) -> dict[str, Any]:
    """Return metadata for a ProtEnc model.

    Args:
        model_name: Model alias such as esm2_t30, esmc_300m, or prot_t5_xl_uniref50.
    """
    import protenc

    return protenc.get_model_info(model_name)


@mcp.tool()
def protenc_embed_sequences(
    sequences: list[str],
    model_name: str | None = None,
    average_sequence: bool = True,
    repr_layer: int | None = None,
    labels: list[str] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Embed one or more protein amino-acid sequences.

    Args:
        sequences: Protein sequences using single-letter amino-acid codes.
        model_name: ProtEnc model alias. Defaults to PROTENC_DEFAULT_MODEL or esm2_t30.
        average_sequence: When true, return one vector per sequence (mean over residues).
        repr_layer: Optional 1-indexed transformer layer. Defaults to the model's final layer.
        labels: Optional names for each sequence. Defaults to seq_0, seq_1, ...
        device: Optional torch device (cuda, cpu, cuda:0). Defaults to PROTENC_DEVICE or auto.
    """
    if not sequences:
        raise ValueError("At least one protein sequence is required.")

    resolved_model = model_name or _default_model()
    resolved_device = device or _default_device()
    resolved_labels = labels or [f"seq_{index}" for index in range(len(sequences))]

    if len(resolved_labels) != len(sequences):
        raise ValueError("labels must have the same length as sequences.")

    encoder = _get_encoder(resolved_model, resolved_device, repr_layer)

    index_to_label = dict(enumerate(resolved_labels))
    results: list[dict[str, Any]] = []
    for seq_index, embedding in encoder.encode(
        sequences,
        average_sequence=average_sequence,
        return_format="numpy",
    ):
        results.append(
            {
                "label": index_to_label[seq_index],
                "sequence_length": len(sequences[seq_index]),
                **_serialize_embedding(embedding),
            }
        )

    return {
        "model_name": resolved_model,
        "device": resolved_device,
        "repr_layer": repr_layer,
        "average_sequence": average_sequence,
        "embeddings": results,
    }


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
