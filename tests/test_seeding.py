from run_pipeline import cell_seed


def test_cell_seed_is_stable_across_processes():
    # Hardcoded, deliberately: a hash()-based implementation would pass a
    # self-consistency check and still give a different stream on every run,
    # silently making a shard unreplayable.
    assert cell_seed("control", "gold", 0) == 535723078
    assert cell_seed("finetuned", "moon", 100) == 2138220600


def test_cell_seed_differs_by_shard():
    same_group = ("control", "gold")

    assert cell_seed(*same_group, 0) != cell_seed(*same_group, 100)


def test_cell_seed_differs_by_organism_word():
    seeds = {
        cell_seed(organism, word, 0)
        for organism in ("control", "prompted")
        for word in ("gold", "moon")
    }

    assert len(seeds) == 4
