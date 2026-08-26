from run_pipeline import cell_seed


def test_cell_seed_is_stable_across_processes():
    # Hardcoded, deliberately: a hash()-based implementation would pass a
    # self-consistency check and still give a different stream on every run,
    # silently making a shard unreplayable.
    assert cell_seed("control", "gold", 12, "pos-11", 0) == 807952643
    assert cell_seed("finetuned", "moon", 0, "assistant_boundary", 100) == 1639112123


def test_cell_seed_differs_by_shard():
    same_cell = ("control", "gold", 12, "pos-11")

    assert cell_seed(*same_cell, 0) != cell_seed(*same_cell, 100)


def test_cell_seed_differs_by_cell():
    seeds = {
        cell_seed(arm, word, layer, position, 0)
        for arm in ("control", "prompted")
        for word in ("gold", "moon")
        for layer in (0, 12)
        for position in ("pos-11", "pos-1")
    }

    assert len(seeds) == 16
