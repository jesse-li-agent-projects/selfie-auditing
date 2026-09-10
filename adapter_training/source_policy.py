"""How much of a source a mixture draws, and by what means.

Kept apart from `grouped_examples` (which imports `torch`) so
`train_adapter`'s argument parsing can name these without paying for the
heavy imports on `--help`.

The two policies are separate types rather than one type with a mode flag, or
a size comparison made at run time: which one a source gets is a property of
that source's population, decided by the caller, and the code paths they
select share nothing but their output type.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Sampled:
    """Draw `weight`'s share of the budget, never repeating a pair.

    For a source whose `(vector, label)` space is far larger than the number
    of examples wanted from it, so rejected redraws stay rare. Cost per
    example climbs sharply once the ask approaches the size of that space --
    at the limit the draw loop is collecting coupons to arrive at the only
    answer the data allows. Use `Exhaustive` there instead.

    :ivar weight: this source's relative share of the sampled budget
    """

    weight: int


@dataclasses.dataclass(frozen=True)
class Exhaustive:
    """Use the source whole: every distinct `(vector, label)` pair, once.

    For a source small enough to exhaust, where the example count is a
    property of the data rather than a number anyone chooses -- so it takes
    no share of the budget and carries no weight.

    Single-topic sources only. A multi-topic group's composed label space is
    the product of its topics' label lists times their orderings, which is
    astronomically large rather than small, so "the whole source" is not a
    thing that can be enumerated (`grouped_examples.enumerate_examples`
    refuses one).
    """


SourcePolicy = Sampled | Exhaustive
