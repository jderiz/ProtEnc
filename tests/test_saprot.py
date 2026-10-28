import pytest
import torch

from protenc import StructureInput, get_model_info, list_models
from protenc.models import SaProtEmbeddingModel
from protenc.saprot import chain_states, combine, sequence_only

from .fixtures import write_backbone_pdb


class TestTokens:
    def test_sequence_only_masks_every_state(self):
        assert sequence_only("MKv") == "M#K#V#"

    def test_combine_interleaves_residues_and_states(self):
        assert combine("MEV", "#vp") == "M#EvVp"

    def test_nonstandard_residues_are_masked_outright(self):
        assert sequence_only("MXB") == "M#" + "##" + "##"
        assert combine("MXV", "vpl") == "Mv" + "##" + "Vl"

    def test_combine_rejects_a_length_mismatch(self):
        with pytest.raises(ValueError, match="3 residues but 2"):
            combine("MEV", "#v")


class TestChainStates:
    @pytest.fixture(autouse=True)
    def _need_mini3di(self):
        pytest.importorskip("mini3di")

    def test_one_letter_per_residue_in_chain_order(self, tmp_path):
        pdb = write_backbone_pdb(tmp_path / "s.pdb", chains={"A": 12, "B": 7})
        assert [len(s) for s in chain_states(pdb)] == [12, 7]
        assert [len(s) for s in chain_states(pdb, ["B", "A"])] == [7, 12]

    def test_unknown_chain_is_reported(self, tmp_path):
        pdb = write_backbone_pdb(tmp_path / "s.pdb", length=8)
        with pytest.raises(ValueError, match=r"Chains \['Z'\]"):
            chain_states(pdb, ["Z"])

    def test_numbering_gap_does_not_add_residues(self, tmp_path):
        pdb = write_backbone_pdb(tmp_path / "s.pdb", length=10, skip={"A": {4}})
        assert [len(s) for s in chain_states(pdb)] == [9]

    def test_ion_named_ca_is_not_a_residue(self, tmp_path):
        pdb = write_backbone_pdb(tmp_path / "s.pdb", length=8)
        text = (
            open(pdb)
            .read()
            .replace(
                "END\n",
                "HETATM 9999 CA    CA A 100      50.000  50.000  50.000  1.00  0.00          CA\nEND\n",
            )
        )
        open(pdb, "w").write(text)
        assert [len(s) for s in chain_states(pdb)] == [8]


class TestModelCards:
    @pytest.mark.parametrize(
        "name, layers, dim",
        [
            ("saprot_35m", 12, 480),
            ("saprot_650m", 33, 1280),
            ("saprot_650m_pdb", 33, 1280),
            ("saprot_1_3b", 66, 1280),
        ],
    )
    def test_cards(self, name, layers, dim):
        info = get_model_info(name)
        assert info["family"] == "SaProt"
        assert (info["num_layers"], info["embed_dim"]) == (layers, dim)
        assert info["structure_input"] == StructureInput.OPTIONAL.value
        assert name in list_models("SaProt")


@pytest.fixture(scope="module")
def saprot_35m():
    try:
        return SaProtEmbeddingModel("westlake-repl/SaProt_35M_AF2", num_layers=12)
    except OSError as e:
        pytest.skip(f"SaProt_35M_AF2 weights unavailable: {e}")


def _embed(model, proteins, **kwargs):
    batch = model.prepare_sequences(proteins, **kwargs)
    return list(model(batch))


class TestEmbedding:
    def test_sequence_only_gives_one_row_per_residue(self, saprot_35m, proteins):
        embeds = _embed(saprot_35m, proteins[:2])
        for prot, embed in zip(proteins, embeds):
            assert embed.shape == (len(prot), 480)

    def test_runs_of_unknown_residues_keep_one_row_each(self, saprot_35m):
        seq = "TVXXSSTKGPSVFPLAPXXXXXXGG"
        (embed,) = _embed(saprot_35m, [seq])
        assert embed.shape == (len(seq), 480)

    def test_padding_does_not_change_an_embedding(self, saprot_35m, proteins):
        alone = _embed(saprot_35m, [proteins[0]])[0]
        padded = _embed(saprot_35m, proteins[:2])[0]
        torch.testing.assert_close(alone, padded, rtol=1e-4, atol=1e-4)

    def test_structure_changes_the_embedding(self, saprot_35m, tmp_path):
        pytest.importorskip("mini3di")
        seq = "MKTVRQERLKSIVRILERSK"
        pdb = write_backbone_pdb(tmp_path / "s.pdb", length=len(seq))
        masked = _embed(saprot_35m, [seq])[0]
        with_structure = _embed(saprot_35m, [seq], structures=pdb)[0]
        assert with_structure.shape == masked.shape
        assert not torch.allclose(masked, with_structure)

    def test_length_mismatch_with_the_structure_is_rejected(self, saprot_35m, tmp_path):
        pytest.importorskip("mini3di")
        pdb = write_backbone_pdb(tmp_path / "s.pdb", length=10)
        with pytest.raises(ValueError, match="do not match the structure"):
            saprot_35m.prepare_sequences(["MKTVRQ"], structures=pdb)

    def test_chain_break_rows_follow_the_esm2_convention(self, saprot_35m):
        # Same as ESM2: the <eos> between chains is kept as a row, residues + 1.
        (embed,) = _embed(saprot_35m, ["MKTV|RQER"])
        assert embed.shape[0] == 4 + 1 + 4
