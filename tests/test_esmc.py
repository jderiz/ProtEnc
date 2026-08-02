"""Tests for ESMC repr_layer support in ProtEnc.

Download weights before running integration tests::

    hf download biohub/ESMC-300M
    hf download biohub/ESMC-600M
    hf download biohub/ESMC-6B
"""
import pytest
import torch

from protenc.models import (
    _esmc_num_layers_from_name,
    get_model,
    get_model_info,
)
from protenc.esmc_loading import ESMC_HF_REPOS, resolve_esmc_repo_id


@pytest.mark.parametrize(
    ("alias", "num_layers", "embed_dim"),
    [
        ("esmc_300m", 30, 960),
        ("esmc_600m", 36, 1152),
        ("esmc_6b", 80, 2560),
    ],
)
def test_esmc_model_info(alias, num_layers, embed_dim):
    info = get_model_info(alias)
    assert info["num_layers"] == num_layers
    assert info["default_repr_layer"] == num_layers
    assert info["embed_dim"] == embed_dim
    assert info["supports_repr_layer"] is True


@pytest.mark.parametrize(
    ("name", "expected_layers"),
    [
        ("esmc_300m", 30),
        ("biohub/ESMC-300M", 30),
        ("esmc_600m", 36),
        ("esmc_6b", 80),
    ],
)
def test_esmc_num_layers_from_name(name, expected_layers):
    assert _esmc_num_layers_from_name(name) == expected_layers


def test_esmc_resolve_repo_id():
    assert resolve_esmc_repo_id("esmc_300m") == ESMC_HF_REPOS["esmc_300m"]
    assert resolve_esmc_repo_id("biohub/ESMC-600M") == "biohub/ESMC-600M"


@pytest.mark.parametrize("alias", ["esmc_300m", "esmc_600m", "esmc_6b"])
def test_esmc_invalid_repr_layer(alias):
    info = get_model_info(alias)
    num_layers = info["num_layers"]
    with pytest.raises(ValueError, match="repr_layer"):
        get_model(alias, repr_layer=num_layers + 1)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("alias", ["esmc_300m"])
def test_esmc_repr_layer_forward(alias):
    """Requires ESMC weights in the HuggingFace cache (hf download biohub/ESMC-300M)."""
    sequence = "MKFL"
    model_default = get_model(alias)
    model_mid = get_model(alias, repr_layer=10)

    batch = model_default.prepare_sequences([sequence])
    out_default = list(model_default(batch))[0]
    out_mid = list(model_mid(batch))[0]

    info = get_model_info(alias)
    assert out_default.shape == (len(sequence), info["embed_dim"])
    assert out_mid.shape == (len(sequence), info["embed_dim"])
    assert model_default.repr_layer == info["num_layers"]
    assert model_mid.repr_layer == 10
    assert not torch.allclose(out_default, out_mid)
