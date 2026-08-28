"""Batching and yield order of `generate_interpretations_batch`, on fakes.

No model, no GPU: what's under test is the bookkeeping around `generate()` --
which row carries which cell's soft token, and when a finished cell is handed
back -- not anything the weights decide.
"""

import torch

from interpret import generate_interpretations_batch

HIDDEN = 4
RESERVED_ID = 7
TEMPLATE_IDS = [1, RESERVED_ID, 2]
N_SAMPLES = 4
N_CELLS = 3


class FakeTokenizer:
    """Tokenizes only SELFIE_TEMPLATE, into TEMPLATE_IDS. `decode` reports the
    soft token a row was given, so a test can tell whose generation it is."""

    eos_token_id = 0

    def __call__(self, text, return_tensors=None, add_special_tokens=None):
        self.input_ids = torch.tensor([TEMPLATE_IDS])
        return self

    def to(self, device):
        return self

    def convert_tokens_to_ids(self, token):
        return RESERVED_ID

    def decode(self, output, skip_special_tokens=False):
        return f"cell {int(output[0])}"


class FakeModel:
    """Records every batch it is asked to generate, and echoes back each row's
    injected soft token as its single output token."""

    def __init__(self):
        self.batch_sizes: list[int] = []

    def get_input_embeddings(self):
        return lambda ids: ids.float().unsqueeze(-1).repeat(1, 1, HIDDEN)

    def generate(self, inputs_embeds, **kwargs):
        self.batch_sizes.append(len(inputs_embeds))
        # Position 1 is the template's only reserved token, so its value is
        # whatever soft token this row injected.
        return inputs_embeds[:, 1, 0].long().unsqueeze(-1)


class FakeAdapter:
    def transform(self, vector):
        return vector


def interpretations(model, batch_size, n_samples=N_SAMPLES):
    # Cell k's soft token is the constant k, so a generation names its cell.
    hidden_vectors = {k: torch.full((HIDDEN,), float(k)) for k in range(N_CELLS)}
    return generate_interpretations_batch(
        model,
        FakeTokenizer(),
        FakeAdapter(),
        hidden_vectors,
        n_samples,
        max_new_tokens=1,
        temperature=0.7,
        device="cpu",
        batch_size=batch_size,
    )


def test_rows_keep_their_own_cells_soft_token():
    """A batch straddling three cells still returns each cell its own rows."""
    cells = dict(interpretations(FakeModel(), batch_size=N_SAMPLES * N_CELLS))

    assert cells == {k: [f"cell {k}"] * N_SAMPLES for k in range(N_CELLS)}


def test_batches_pool_across_cells():
    model = FakeModel()

    dict(interpretations(model, batch_size=5))

    # 3 cells x 4 samples = 12 rows, so pooling is what allows a batch of 5.
    assert model.batch_sizes == [5, 5, 2]


def test_cells_arrive_before_the_group_finishes():
    """Each cell is yielded as its last row is decoded -- what lets a caller
    write a cell to disk mid-group rather than after the whole group."""
    model = FakeModel()

    # With batch_size=5, cell k's last row falls in batch k (0-indexed).
    for expected_key, (key, _) in enumerate(interpretations(model, batch_size=5)):
        assert key == expected_key
        assert len(model.batch_sizes) == expected_key + 1


def test_nothing_runs_until_iterated():
    """Being a generator, the call itself must not consume the RNG stream a
    caller seeds -- otherwise seeding before iteration would come too late."""
    model = FakeModel()

    interpretations(model, batch_size=5)

    assert model.batch_sizes == []
