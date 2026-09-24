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
    # DataParallel requires the module on device_ids[0] (cuda:0) when CUDA is visible.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = get_model("esm2_t6").to(device)
    batch = model.prepare_sequences(SEQS)
    expected = list(model(batch))

    model.enable_data_parallel()

    assert isinstance(model._core, nn.DataParallel)
    assert not isinstance(model.model, nn.DataParallel)
    assert "_core" not in dict(model.named_children())
    for got, exp in zip(model(batch), expected):
        assert torch.allclose(got, exp, atol=1e-4)


def test_drop_self_bound_forwards_after_unwrapped_call():
    """An unwrapped HF forward leaves self-bound ``forward`` attributes on layers
    (transformers output capturing); they must be removed before replication, or
    DataParallel replicas call the original module on cuda:0."""
    from protenc.models import _drop_self_bound_forwards

    model = get_model("esm2_t6")
    list(model(model.prepare_sequences(SEQS)))
    assert any("forward" in m.__dict__ for m in model.model.modules())

    _drop_self_bound_forwards(model.model)

    assert not any("forward" in m.__dict__ for m in model.model.modules())
    list(model(model.prepare_sequences(SEQS)))  # still runs


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


def _assert_embeddings_match(got, exp, bf16: bool, name: str):
    """fp32: elementwise allclose. bf16: bf16 rounding differs with the per-GPU batch
    size (kernel choice) and grows over depth, and a few large-magnitude dimensions make
    an absolute tolerance meaningless, so compare direction and relative error instead."""
    assert got.shape == exp.shape, f"{name}: shape {got.shape} != {exp.shape}"
    if not bf16:
        max_diff = (got - exp).abs().max().item()
        assert torch.allclose(got, exp, atol=1e-4), f"{name}: max |diff| {max_diff:.2e}"
        return
    got, exp = got.float(), exp.float()
    cos = torch.nn.functional.cosine_similarity(got, exp, dim=-1)  # per residue
    rel = ((got - exp).norm() / exp.norm()).item()
    assert cos.min().item() > 0.99, f"{name}: min per-residue cosine {cos.min():.4f}"
    assert rel < 5e-2, f"{name}: relative L2 error {rel:.3e}"


@requires_multi_gpu
@pytest.mark.parametrize(
    "model_name, seqs, bf16",
    [
        ("esm2_t6", SEQS, False),
        ("esmc_300m", SEQS, True),  # weights cast to bf16 on CUDA
        ("esm3", SAME_LEN_SEQS, True),  # bf16 autocast
    ],
)
def test_data_parallel_matches_single_gpu(model_name, seqs, bf16):
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
        _assert_embeddings_match(got[idx], expected[idx], bf16, f"{model_name}[{idx}]")

    layers = [1, parallel.repr_layer]
    multi = list(
        parallel.encode_multi_layer(
            seqs, layers, average_sequence=False, show_progress=False
        )
    )
    assert len(multi) == len(seqs) * len(layers)
    for idx, layer, emb in multi:
        if layer == parallel.repr_layer:
            _assert_embeddings_match(
                emb, expected[idx], bf16, f"{model_name}[{idx}] multi-layer"
            )
