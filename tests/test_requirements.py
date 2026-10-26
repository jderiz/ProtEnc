import pytest

from protenc import EncoderRequirements, StructureInput
from protenc.encoder import ProteinEncoder
from protenc.models import (
    EmbeddingType,
    ProteinMPNNEmbeddingModel,
    get_model_info,
    list_models,
)


class _StubModel:
    embedding_kind = EmbeddingType.PER_RESIDUE

    def __init__(self, structure_input, requires_same_length_batch=False):
        self.structure_input = structure_input
        self.requires_same_length_batch = requires_same_length_batch


class TestModelInfoAdvertisesStructureInput:
    @pytest.mark.parametrize("model_name", list_models())
    def test_value_is_a_known_structure_input(self, model_name):
        value = get_model_info(model_name)["structure_input"]
        assert value in {s.value for s in StructureInput}

    @pytest.mark.parametrize(
        "model_name, expected",
        [("mpnn", "required"), ("esm3", "optional"), ("esm2_t6", "none")],
    )
    def test_known_models(self, model_name, expected):
        assert get_model_info(model_name)["structure_input"] == expected


class TestEncoderRequirements:
    def test_structure_is_taken_from_the_model(self):
        encoder = ProteinEncoder(_StubModel(StructureInput.REQUIRED))
        assert encoder.requirements.structure is StructureInput.REQUIRED

    def test_same_length_sequences_is_taken_from_the_model(self):
        encoder = ProteinEncoder(_StubModel(StructureInput.NONE, True))
        assert encoder.requirements == EncoderRequirements(
            structure=StructureInput.NONE, same_length_sequences=True
        )

    def test_model_without_the_attribute_needs_nothing(self):
        class Plain:
            embedding_kind = EmbeddingType.PER_RESIDUE

        assert ProteinEncoder(Plain()).requirements == EncoderRequirements()

    def test_mpnn_needs_one_structure_with_matching_lengths(self):
        assert ProteinMPNNEmbeddingModel.structure_input is StructureInput.REQUIRED
        assert ProteinMPNNEmbeddingModel.requires_same_length_batch is True

    def test_encoding_without_a_required_structure_fails_before_any_work(self):
        encoder = ProteinEncoder(_StubModel(StructureInput.REQUIRED))
        with pytest.raises(ValueError, match="requires structures"):
            list(encoder.encode(["ACDE"]))

    def test_multi_layer_encoding_checks_too(self):
        encoder = ProteinEncoder(_StubModel(StructureInput.REQUIRED))
        with pytest.raises(ValueError, match="requires structures"):
            list(encoder.encode_multi_layer(["ACDE"], [1]))

    def test_optional_structure_may_be_omitted(self):
        encoder = ProteinEncoder(_StubModel(StructureInput.OPTIONAL))
        encoder._check_requirements(None)
