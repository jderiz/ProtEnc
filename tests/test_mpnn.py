import numpy as np
import pytest
import torch
import torch.nn.functional as F

from protenc import models
from protenc.mpnn import ProteinMPNN
from protenc.models import ProteinMPNNEmbeddingModel, _mpnn_core, _mpnn_encode

from .fixtures import write_backbone_pdb

L = 12
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _build(monkeypatch, decoding):
    torch.manual_seed(0)
    net = ProteinMPNN(
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        k_neighbors=8,
        augment_eps=0.0,
    ).eval()
    monkeypatch.setattr(
        ProteinMPNN, "from_pretrained", classmethod(lambda cls, *a, **k: net)
    )
    return ProteinMPNNEmbeddingModel("v_48_020.pt", decoding=decoding), net


def _embed(model, sequences, pdb):
    batch = model.prepare_sequences(sequences, pdb)
    return torch.stack(list(model(batch)))


@pytest.fixture
def pdb(tmp_path):
    return write_backbone_pdb(tmp_path / "bb.pdb", L)


@pytest.fixture
def sequences():
    rng = np.random.default_rng(1)
    base = "".join(rng.choice(list(ALPHABET), size=L))
    mutated = base[:5] + ("A" if base[5] != "A" else "C") + base[6:]
    return base, mutated


class TestMpnnEmbedding:
    def test_encoder_ignores_the_sequence(self, monkeypatch, pdb, sequences):
        model, net = _build(monkeypatch, "left_to_right")
        feats = []
        for seq in sequences:
            batch = model.prepare_sequences([seq], pdb)
            feats.append(_mpnn_encode(net, batch)[0])
        assert torch.equal(feats[0], feats[1])

    @pytest.mark.parametrize("decoding", ["left_to_right", "leave_one_out"])
    def test_embedding_depends_on_the_sequence(
        self, monkeypatch, pdb, sequences, decoding
    ):
        model, _ = _build(monkeypatch, decoding)
        emb = _embed(model, list(sequences), pdb)
        assert emb.shape == (2, L, 128)
        assert not torch.allclose(emb[0], emb[1])

    def test_left_to_right_matches_upstream_forward(self, monkeypatch, pdb, sequences):
        model, net = _build(monkeypatch, "left_to_right")
        batch = model.prepare_sequences([sequences[0]], pdb)
        ours = _mpnn_core(net, batch, (0,), decoding="left_to_right")[:, 0]
        upstream = net(
            batch["X"],
            batch["S"],
            batch["mask"],
            batch["chain_M"],
            batch["residue_idx"],
            batch["chain_encoding_all"],
            torch.zeros(1, L),
            use_input_decoding_order=True,
            decoding_order=torch.arange(L)[None],
        )
        assert torch.allclose(
            F.log_softmax(net.W_out(ours), dim=-1), upstream, atol=1e-5
        )

    def test_left_to_right_first_position_sees_no_sequence(
        self, monkeypatch, pdb, sequences
    ):
        model, _ = _build(monkeypatch, "left_to_right")
        emb = _embed(model, list(sequences), pdb)
        assert torch.allclose(emb[0, 0], emb[1, 0], atol=1e-6)

    def test_leave_one_out_masks_only_the_residue_itself(
        self, monkeypatch, pdb, sequences
    ):
        model, _ = _build(monkeypatch, "leave_one_out")
        emb = _embed(model, list(sequences), pdb)
        # The sequences differ only at position 5.
        assert torch.allclose(emb[0, 5], emb[1, 5], atol=1e-6)
        others = [i for i in range(L) if i != 5]
        assert not torch.allclose(emb[0, others], emb[1, others])

    def test_rejects_unknown_decoding(self, monkeypatch):
        with pytest.raises(ValueError, match="decoding"):
            _build(monkeypatch, "random")
