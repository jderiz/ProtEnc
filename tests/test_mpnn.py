import numpy as np
import pytest
import torch

from protenc.models import ProteinMPNNEmbeddingModel, _mpnn_encode
from protenc.mpnn import ProteinMPNN

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


def _seq(n, shift=0):
    return "".join(ALPHABET[(i + shift) % len(ALPHABET)] for i in range(n))


class TestMpnnStructureLayout:
    """Gaps in the residue numbering, chain breaks and chain order."""

    def test_numbering_gap_rows_are_not_residues(self, monkeypatch, tmp_path):
        pdb = write_backbone_pdb(tmp_path / "gap.pdb", L, skip={"A": {4, 5}})
        model, _ = _build(monkeypatch)
        batch = model.prepare_sequences([_seq(L - 2)], pdb)
        # The filler rows stay in the tensors, masked, so residue_idx spacing is
        # upstream's; the output has residues only.
        assert batch["X"].shape[1] == L
        assert batch["mask"][0, 3:5].tolist() == [0.0, 0.0]
        (emb,) = list(model(batch))
        assert emb.shape == (L - 2, 128)

    def test_residues_next_to_a_gap_see_the_gap_in_their_spacing(
        self, monkeypatch, tmp_path
    ):
        pdb = write_backbone_pdb(tmp_path / "gap.pdb", L, skip={"A": {4, 5}})
        model, _ = _build(monkeypatch)
        batch = model.prepare_sequences([_seq(L - 2)], pdb)
        idx = batch["residue_idx"][0]
        assert (idx[5] - idx[2]).item() == 3  # residues 3 and 6 are three apart

    def test_sequence_that_counts_the_gap_is_rejected(self, monkeypatch, tmp_path):
        pdb = write_backbone_pdb(tmp_path / "gap.pdb", L, skip={"A": {4, 5}})
        model, _ = _build(monkeypatch)
        with pytest.raises(ValueError, match="Chain A.*numbering gaps excluded"):
            model.prepare_sequences([_seq(L)], pdb)

    def test_chain_breaks_split_the_sequence(self, monkeypatch, tmp_path):
        pdb = write_backbone_pdb(
            tmp_path / "two.pdb", chains={"A": 8, "B": 6}, skip={"B": {2}}
        )
        model, _ = _build(monkeypatch)
        seq = _seq(8) + "|" + _seq(5, 3)
        (emb,) = list(model(model.prepare_sequences([seq], pdb)))
        assert emb.shape == (13, 128)

    def test_chains_without_a_separator_are_cut_by_residue_count(
        self, monkeypatch, tmp_path
    ):
        pdb = write_backbone_pdb(tmp_path / "two.pdb", chains={"A": 8, "B": 6})
        model, _ = _build(monkeypatch)
        a, b = _seq(8), _seq(6, 3)
        with_break = _embed(model, [a + "|" + b], pdb)
        without = _embed(model, [a + b], pdb)
        assert torch.allclose(with_break, without)

    def test_rows_follow_chain_list_not_the_alphabet(self, monkeypatch, tmp_path):
        pdb = write_backbone_pdb(tmp_path / "two.pdb", chains={"B": 6, "A": 5})
        model, _ = _build(monkeypatch)
        a, b = _seq(5), _seq(6, 3)
        ab = list(model(model.prepare_sequences([a + "|" + b], pdb, ["A", "B"])))[0]
        ba = list(model(model.prepare_sequences([b + "|" + a], pdb, ["B", "A"])))[0]
        assert torch.allclose(ab[:5], ba[6:], atol=1e-5)  # chain A
        assert torch.allclose(ab[5:], ba[:6], atol=1e-5)  # chain B

    def test_wrong_number_of_chain_segments_is_rejected(self, monkeypatch, tmp_path):
        pdb = write_backbone_pdb(tmp_path / "two.pdb", chains={"A": 8, "B": 6})
        model, _ = _build(monkeypatch)
        with pytest.raises(ValueError, match="3 chain segments"):
            model.prepare_sequences([_seq(4) + "|" + _seq(4) + "|" + _seq(6)], pdb)

    def test_chain_list_restricts_the_chains(self, monkeypatch, tmp_path):
        pdb = write_backbone_pdb(tmp_path / "two.pdb", chains={"A": 8, "B": 6})
        model, _ = _build(monkeypatch)
        (emb,) = list(model(model.prepare_sequences([_seq(6)], pdb, ["B"])))
        assert emb.shape == (6, 128)
