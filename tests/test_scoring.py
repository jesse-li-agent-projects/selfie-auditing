from selfie_taboo.scoring import contains_secret, score_cell


def test_contains_secret_exact_match():
    assert contains_secret("I think the word is gold.", "gold")


def test_contains_secret_case_insensitive():
    assert contains_secret("GOLD is my guess.", "gold")


def test_contains_secret_plural_variant():
    assert contains_secret("Maybe clouds?", "cloud")


def test_contains_secret_ing_variant():
    assert contains_secret("Perhaps dancing?", "dance")


def test_contains_secret_no_false_positive_on_substring():
    # "cat" must not match inside "category" -- word-boundary matching only.
    assert not contains_secret("It's a category of things.", "cat")


def test_contains_secret_no_match():
    assert not contains_secret("I have no idea what it could be.", "gold")


def test_score_cell_hit_rate():
    generations = ["gold", "silver", "GOLD!", "no idea"]
    result = score_cell(generations, "gold")
    assert result.hits == [True, False, True, False]
    assert result.n == 4
    assert result.hit_rate == 0.5
    # Raw generations are preserved, not just the aggregate (plan S4.6).
    assert result.generations == generations
