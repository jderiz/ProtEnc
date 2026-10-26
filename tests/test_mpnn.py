import numpy as np
import pytest
import torch

from protenc.mpnn import ProteinMPNN
from protenc.models import ProteinMPNNEmbeddingModel, _mpnn_encode

from .fixtures import write_backbone_pdb

L = 12
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _build(monkeypatch):
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
    return ProteinMPNNEmbeddingModel("v_48_020.pt"), net


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
        model, net = _build(monkeypatch)
        feats = []
        for seq in sequences:
            batch = model.prepare_sequences([seq], pdb)
            feats.append(_mpnn_encode(net, batch)[0])
        assert torch.equal(feats[0], feats[1])

    def test_embedding_depends_on_the_sequence(self, monkeypatch, pdb, sequences):
        model, _ = _build(monkeypatch)
        emb = _embed(model, list(sequences), pdb)
        assert emb.shape == (2, L, 128)
        assert not torch.allclose(emb[0], emb[1])

    def test_mutated_position_sees_its_own_residue(self, monkeypatch, pdb, sequences):
        model, _ = _build(monkeypatch)
        emb = _embed(model, list(sequences), pdb)
        # The sequences differ only at position 5.
        assert not torch.allclose(emb[0, 5], emb[1, 5])

    def test_batching_does_not_change_the_embedding(self, monkeypatch, pdb, sequences):
        model, _ = _build(monkeypatch)
        together = _embed(model, list(sequences), pdb)
        alone = torch.cat([_embed(model, [seq], pdb) for seq in sequences])
        assert torch.allclose(together, alone, atol=1e-5)
