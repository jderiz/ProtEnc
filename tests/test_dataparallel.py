import warnings
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from protenc.console.extract import _maybe_apply_data_parallel
from protenc.encoder import ProteinEncoder, get_encoder
from protenc.models import get_model
from tests.utils import skip_no_gpu


def test_esmc_data_parallel_disabled_on_get_encoder():
    """ESMC models should not use DataParallel even when requested."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    with patch("protenc.encoder.is_esmc_model", return_value=True):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            encoder = get_encoder("prot_bert", device="cuda", data_parallel=True)

    assert encoder.data_parallel is False
    assert not isinstance(encoder.model.model, nn.DataParallel)
    assert any(
        "DataParallel is not supported for ESMC models" in str(w.message)
        for w in caught
    )


def test_esmc_data_parallel_disabled_in_extract():
    """extract.py should skip DataParallel for ESMC models."""
    model = get_model("prot_bert")
    inner = model.model

    with patch("protenc.console.extract.is_esmc_model", return_value=True):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _maybe_apply_data_parallel(model, data_parallel=True, device_ids=None)

    assert model.model is inner
    assert not isinstance(model.model, nn.DataParallel)
    assert any(
        "DataParallel is not supported for ESMC models" in str(w.message)
        for w in caught
    )


@pytest.mark.parametrize("device", ["cuda"])
@skip_no_gpu
def test_data_parallel_info_and_validation(device):
    """validate_data_parallel_setup and get_data_parallel_info on a small model."""
    encoder = ProteinEncoder(get_model("prot_bert"), data_parallel=True)
    encoder.model = encoder.model.to(device)

    info = encoder.get_data_parallel_info()
    issues = encoder.validate_data_parallel_setup()

    if torch.cuda.device_count() < 2:
        assert info["enabled"] is True
        assert info["device_count"] == 1
        assert any("only one GPU" in issue for issue in issues)
    else:
        assert info["enabled"] is True
        assert info["device_count"] >= 2
        assert "primary_device" in info
        assert issues == []
