import torch
from selfie_adapters import load_adapter

from smoke.small_llama_config import create_random_selfie_adapter, embedding_norm

HIDDEN_DIM = 64
INIT_SCALE = 1.5


def test_random_adapter_round_trips_through_the_real_loader(tmp_path):
    """The generated checkpoint has to satisfy selfie_adapters' own loader --
    that is the whole point of writing a file instead of a stub object."""
    path = create_random_selfie_adapter(
        HIDDEN_DIM, tmp_path / "adapter.safetensors", INIT_SCALE
    )

    adapter = load_adapter(str(path), device="cpu")

    assert adapter.model_dim == HIDDEN_DIM
    assert adapter.get_metadata()["projection_type"] == "scalar_affine"
    soft_token = adapter.transform(torch.randn(HIDDEN_DIM))
    assert soft_token.shape == (HIDDEN_DIM,)


def test_soft_token_lands_near_the_requested_scale(tmp_path):
    """A soft token far outside embedding scale degenerates generation for
    reasons unrelated to the shapes this checkpoint exists to test."""
    path = create_random_selfie_adapter(
        HIDDEN_DIM, tmp_path / "adapter.safetensors", INIT_SCALE
    )
    adapter = load_adapter(str(path), device="cpu")

    norm = adapter.transform(torch.randn(HIDDEN_DIM)).norm().item()

    assert INIT_SCALE * 0.8 < norm < INIT_SCALE * 1.2


def test_embedding_norm_reads_the_models_own_embedding_table():
    class FakeModel:
        def get_input_embeddings(self):
            embedding = torch.nn.Embedding(4, HIDDEN_DIM)
            with torch.no_grad():
                embedding.weight.fill_(0.5)
            return embedding

    expected = 0.5 * HIDDEN_DIM**0.5
    assert embedding_norm(FakeModel()) == torch.tensor(expected).item()
