import pytest
import torch

from protenc.models import get_model, get_model_info
from protenc.encoder import get_encoder
from .utils import list_models_to_test, skip_no_gpu, skip_large_models


@pytest.mark.parametrize('model_name', list_models_to_test())
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@skip_no_gpu
@skip_large_models(max_embed_dim=1280)
def test_get_encoder(model_name, device):
    encoder = get_encoder(model_name, device=device)
    model = get_model(model_name).to(device)

    for encoder_param, model_param in zip(encoder.model.parameters(), model.parameters()):
        assert torch.allclose(encoder_param, model_param)
        assert encoder_param.device.type == device


@skip_no_gpu
@skip_large_models(max_embed_dim=1280)
@pytest.mark.parametrize('model_name', list_models_to_test())
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_encode(proteins, model_name, device):
    model_info = get_model_info(model_name)
    encoder = get_encoder(model_name, device=device)

    for idx, embed in encoder(proteins, average_sequence=False):
        assert len(proteins[idx]) == len(embed)
        assert embed.shape[-1] == model_info['embed_dim']


def test_encode_list_yields_index_embed_tuples():
    """List encode yields (index, embedding) pairs without loading a model or GPU."""
    from unittest.mock import MagicMock

    from protenc.encoder import ProteinEncoder

    proteins = ['ACDE', 'FGHI', 'KLMN']
    mock_embeds = [torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8)]

    def fake_encode_batches(*_args, **_kwargs):
        for i, emb in enumerate(mock_embeds):
            yield i, emb

    encoder = ProteinEncoder(MagicMock())
    encoder._encode_batches = fake_encode_batches

    for idx, embed in encoder(proteins):
        assert isinstance(idx, int)
        assert 0 <= idx < len(proteins)
        assert torch.equal(embed, mock_embeds[idx])


@skip_no_gpu
@skip_large_models(max_embed_dim=1280)
@pytest.mark.parametrize('model_name', list_models_to_test())
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_encode_dict(protein_dict, model_name, device):
    model_info = get_model_info(model_name)
    encoder = get_encoder(model_name, device=device)

    for prot_id, embed in encoder(protein_dict, average_sequence=False):
        assert len(protein_dict[prot_id]) == len(embed)
        assert embed.shape[-1] == model_info['embed_dim']
