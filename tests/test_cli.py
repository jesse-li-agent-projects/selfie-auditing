"""run_pipeline.py's flag parsing.

No heavy imports: parsing happens before anything touches the Hub or loads
weights, which is the point -- a typo has to fail here, not minutes later.
"""

import pytest

from config import ModelOrganism, Position
from run_pipeline import parse_args

BASE = ["--words", "gold", "--output-dir", "out"]


def parse(*flags):
    return parse_args(BASE + list(flags))


def test_defaults_leave_the_config_to_decide():
    args = parse()

    assert args.organisms is None and args.positions is None
    assert args.layers == "all"


def test_arms_and_positions_parse_to_their_enums():
    args = parse("--organisms", "control,finetuned", "--positions", "assistant_boundary,-1")

    assert args.organisms == [ModelOrganism.CONTROL, ModelOrganism.FINETUNED]
    assert args.positions == [Position.ASSISTANT_BOUNDARY, -1]


@pytest.mark.parametrize(
    "flags",
    [
        ("--organisms", "controll"),
        ("--positions", "assistant_bounadry"),
        ("--layers", "0,4,x"),
    ],
)
def test_a_bad_value_fails_at_parse_time(flags, capsys):
    # Before this, these raised out of main() -- after the output directory was
    # created and after a Hub round trip for the base model's config.
    with pytest.raises(SystemExit):
        parse(*flags)

    assert "usage:" in capsys.readouterr().err
