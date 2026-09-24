import pytest
import torch
import torch.nn as nn

from protenc.console.extract import _create_encoder
from protenc.encoder import ProteinEncoder, get_encoder
from protenc.models import get_model
from protenc.utils import NestedNamespace
from tests.utils import skip_no_gpu

SEQS = ["MKTAYIAKQRQISFVKSHFSRQ", "MKTAYIAKQ", "GSHMLEDPVDAFQ", "MPLLLLAAAGGG"]
# Same-length sequences for models that require same-length batches (ESM3).
SAME_LEN_SEQS = ["MKTAYIAKQRQI", "GSHMLEDPVDAF", "MPLLLLAAAGGG", "MKTAYIAKQAAA"]

requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires at least two GPUs",
)


def test_core_output_is_batch_first():
    """The compute core returns [B, n_layers, L, D] so DataParallel gathers on dim 0."""
    model = get_model("esm2_t6")
    batch = model.prepare_sequences(SEQS)
    hidden = model._run_core(batch, (1, model.repr_layer))

    assert hidden.shape[:2] == (len(SEQS), 2)
    assert hidden.shape[2] == batch["input_ids"].shape[1]


def test_enable_data_parallel_keeps_backbone_unwrapped():
    """Only the compute core is wrapped; outputs and attribute access are unchanged."""
    model = get_model("esm2_t6")
    batch = model.prepare_sequences(SEQS)
    expected = list(model(batch))

    model.enable_data_parallel()

    assert isinstance(model._core, nn.DataParallel)
    assert not isinstance(model.model, nn.DataParallel)
    assert "_core" not in dict(model.named_children())
    for got, exp in zip(model(batch), expected):
        assert torch.allclose(got, exp)


@skip_no_gpu
@pytest.mark.parametrize("device", ["cuda"])
def test_esmc_data_parallel_enabled_on_get_encoder(device):
    """ESMC models are replicated with DataParallel like every other embedder."""
    encoder = get_encoder("esmc_300m", device=device, data_parallel=True)

    assert encoder.data_parallel is True
    assert isinstance(encoder.model._core, nn.DataParallel)
    assert not isinstance(encoder.model.model, nn.DataParallel)


@skip_no_gpu
@pytest.mark.parametrize("device", ["cuda"])
def test_data_parallel_via_cli_encoder_factory(device):
    """CLI _create_encoder wraps the compute core when --data_parallel is set."""
    args = NestedNamespace(
        model_name="esm2_t6",
        repr_layer=None,
        device=device,
        data_parallel=True,
        device_ids=None,
        batch_size=64,
        amp=False,
        num_workers=0,
    )
    encoder = _create_encoder(args)

    assert encoder.data_parallel is True
    assert isinstance(encoder.model._core, nn.DataParallel)


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


@requires_multi_gpu
@pytest.mark.parametrize(
    "model_name, seqs, atol",
    [
        ("esm2_t6", SEQS, 1e-4),
        ("esmc_300m", SEQS, 1e-3),
        ("esm3", SAME_LEN_SEQS, 5e-2),  # bf16 autocast
    ],
)
def test_data_parallel_matches_single_gpu(model_name, seqs, atol):
    """Multi-GPU embeddings equal single-GPU embeddings for each family."""
    single = get_encoder(model_name, device="cuda:0", batch_size=len(seqs))
    expected = dict(single.encode(seqs, average_sequence=False, show_progress=False))
    del single
    torch.cuda.empty_cache()

    parallel = get_encoder(model_name, data_parallel=True, batch_size=len(seqs))
    assert isinstance(parallel.model._core, nn.DataParallel)
    got = dict(parallel.encode(seqs, average_sequence=False, show_progress=False))

    assert got.keys() == expected.keys()
    for idx in expected:
        assert torch.allclose(got[idx], expected[idx], atol=atol)

    layers = [1, parallel.repr_layer]
    multi = list(
        parallel.encode_multi_layer(
            seqs, layers, average_sequence=False, show_progress=False
        )
    )
    assert len(multi) == len(seqs) * len(layers)
    for idx, layer, emb in multi:
        if layer == parallel.repr_layer:
            assert torch.allclose(emb, expected[idx], atol=atol)
