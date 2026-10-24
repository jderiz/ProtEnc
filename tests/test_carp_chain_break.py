import torch
from sequence_models.constants import PROTEIN_ALPHABET

from protenc.models import CarpEmbeddingModel


class _StubCollater:
    """Stands in for CARP's collater, which needs the real checkpoint to build."""

    pad_idx = PROTEIN_ALPHABET.index("-")


class TestCarpChainBreakToken:
    """CARP's alphabet has no chain-break character, so it declares GAP instead.

    GAP and PAD are the same character in CARP, which makes a mid-sequence break
    indistinguishable from padding; these pin both halves of that fix.
    """

    def test_carp_declares_gap_as_its_chain_break_token(self):
        # Not the canonical "|": PROTEIN_ALPHABET has no chain-break character, and
        # sequence_models defines SEP ("/") only inside MSA_ALPHABET_PLUS.
        assert CarpEmbeddingModel.chain_break_token == "-"

    def test_chain_break_token_is_in_the_protein_alphabet(self):
        assert CarpEmbeddingModel.chain_break_token in PROTEIN_ALPHABET

    def test_recorded_lengths_survive_a_mid_sequence_break(self):
        model = CarpEmbeddingModel.__new__(CarpEmbeddingModel)
        model.collater = _StubCollater()

        # "ACDE-FGHIK": ten characters, the fifth being the chain break, which
        # tokenizes to pad_idx.
        pad = _StubCollater.pad_idx
        tokens = torch.tensor([[0, 2, 3, 4, pad, 5, 6, 7, 8, 9]])

        recorded = model._sequence_lengths(
            {"tokens": tokens, "lengths": torch.tensor([10])}
        )
        assert int(recorded[0]) == 10

        # Without the recorded lengths the break counts as padding, so the slice
        # that follows would drop the final residue.
        inferred = model._sequence_lengths({"tokens": tokens})
        assert int(inferred[0]) == 9
