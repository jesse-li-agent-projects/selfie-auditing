from pathlib import Path

from config import (
    TABOO_WORDS,
    Arm,
    Position,
    full_sweep_config,
    layers_full,
    layers_smoke,
)


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


def test_full_sweep_config():
    config = full_sweep_config(
        ["gold", "moon"],
        num_hidden_layers=32,
        output_dir=Path("out"),
        n_samples=50,
        sample_start=100,
        device="cuda:3",
    )

    assert config.layers == layers_full(32)
    assert config.positions == [Position.FULL_USER_SPAN]
    assert config.arms == [Arm.CONTROL, Arm.PROMPTED, Arm.FINETUNED]
    assert config.words == ["gold", "moon"]
    assert config.n_samples == 50
    # Both of these reaching the config is the point. The superseded
    # first_pass_config never set `device`, so extraction and generation
    # targeted the default GPU no matter what --device said.
    assert config.device == "cuda:3"
    assert config.sample_start == 100
