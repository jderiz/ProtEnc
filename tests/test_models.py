import pytest
import torch

from protenc.models import get_model, get_model_info

from .utils import list_models_to_test, model_inputs, skip_no_gpu


@skip_no_gpu
@pytest.mark.parametrize("model_name", list_models_to_test())
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_protein_language_model(model_name, proteins, device, tmp_path):
    model = get_model(model_name).to(device)
    model_info = get_model_info(model_name)
    proteins, structures = model_inputs(model_name, proteins, tmp_path)

    if structures is None:
        batch = model.prepare_sequences(proteins)
    else:
        batch = model.prepare_sequences(proteins, structures)
    if isinstance(batch, dict):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    else:
        batch = batch.to(device)
    embeds = list(model(batch))

    for prot, embed in zip(proteins, embeds):
        assert len(prot) == len(embed)
        assert embed.shape[-1] == model_info["embed_dim"]
