from pathlib import Path

from config import TABOO_WORDS, ModelOrganism, Position, resolve_layers, sweep_config


def test_resolve_layers_all_means_every_layer():
    assert resolve_layers("all", 32) == list(range(32))


def test_resolve_layers_parses_a_comma_list():
    assert resolve_layers("0,4,8,12", 32) == [0, 4, 8, 12]


def test_taboo_words_count():
    # research_notes_selfie_mechanism.md S3: 20 secret-word variants.
    assert len(TABOO_WORDS) == 20
    assert len(set(TABOO_WORDS)) == 20  # no duplicates


def test_sweep_config_defaults_to_the_real_8b_run():
    config = sweep_config(
        ["gold", "moon"],
        layers=list(range(32)),
        output_dir=Path("out"),
        n_samples=50,
        sample_start=100,
        device="cuda:3",
        batch_size=8,
    )

    assert config.layers == list(range(32))
    assert config.positions == [Position.USER_PROMPT_SPAN]
    assert config.organisms == [ModelOrganism.CONTROL, ModelOrganism.PROMPTED, ModelOrganism.FINETUNED]
    assert config.words == ["gold", "moon"]
    assert config.n_samples == 50
    # Both of these reaching the config is the point. The superseded
    # first_pass_config never set `device`, so extraction and generation
    # targeted the default GPU no matter what --device said.
    assert config.device == "cuda:3"
    assert config.sample_start == 100
    assert config.batch_size == 8


def test_sweep_config_overrides_arms_and_positions():
    config = sweep_config(
        ["gold"],
        layers=[0, 8],
        organisms=[ModelOrganism.CONTROL],
        positions=[Position.ASSISTANT_BOUNDARY, -1],
        output_dir=Path("out"),
    )

    assert config.organisms == [ModelOrganism.CONTROL]
    assert config.positions == [Position.ASSISTANT_BOUNDARY, -1]
