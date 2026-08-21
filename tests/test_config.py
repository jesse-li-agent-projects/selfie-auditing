from selfie_taboo.config import TABOO_WORDS, layers_full, layers_smoke


def test_layers_smoke_matches_plan_example():
    # Plan S4.4: "every 4th layer ... 8 layers at N = 32".
    assert layers_smoke(32) == [0, 4, 8, 12, 16, 20, 24, 28]
    assert len(layers_smoke(32)) == 8


def test_layers_full_matches_plan_example():
    assert layers_full(32) == list(range(32))
    assert len(layers_full(32)) == 32


def test_taboo_words_count():
    # research_notes_selfie_mechanism.md S3: 20 secret-word variants.
    assert len(TABOO_WORDS) == 20
    assert len(set(TABOO_WORDS)) == 20  # no duplicates
