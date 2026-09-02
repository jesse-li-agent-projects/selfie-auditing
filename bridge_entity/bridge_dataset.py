"""TwoHopFact questions for the bridge-entity experiment (SelfIE adapter paper S3.6).

A two-hop question ("The author of the novel Nineteen Eighty-Four was born in
the city of") has an intermediate answer -- the *bridge entity*, here George
Orwell -- that appears in neither the question nor the model's reply. The
experiment asks whether SelfIE can read that entity out of the model's
activations anyway.

That question is only meaningful where the model really does route through the
bridge, so the dataset is filtered to questions the model answers correctly
*and* whose first hop it can answer on its own (`filter_bridge_questions.py`
runs the model; the two prompts it asks with live here). A question then
carries its own answer keys: an entity is matched by any of its Wikidata
aliases, since a generation may use any of an entity's names.

Kept free of heavy imports so reading the dataset costs no torch import.
"""

from __future__ import annotations

import ast
import csv
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from prompts import (
    BRIDGE_FILTER_PROMPT,
    BRIDGE_FILTER_SHOT_ANSWER,
    BRIDGE_FILTER_SHOT_BRIDGE,
)

# TwoHopFact names entities e1 -> e2 -> e3 along the reasoning chain. This
# module renames them for what they are in the experiment: e2 is the bridge,
# e3 the final answer, and e1 only ever appears inside the question text.
_COLUMNS = {
    "id": "uid",
    "statement": "r2(r1(e1)).prompt",
    "bridge_statement": "r1(e1).prompt",
    "bridge_value": "e2.value",
    "bridge_category": "e2.rough_category",
    "answer_value": "e3.value",
    "answer_category": "e3.rough_category",
    "category": "category",
    "fact_comp_type": "fact_comp_type",
}

# The dataset's own category vocabulary, as the reference filter renders it in
# a prompt: "the name of a human" reads better than "the name of a person".
_DISPLAY_CATEGORY = {"person": "human"}


@dataclass(frozen=True)
class BridgeQuestion:
    """One two-hop question, its bridge entity, and its final answer."""

    id: str
    statement: str
    bridge_statement: str
    bridge_value: str
    bridge_aliases: tuple[str, ...]
    bridge_category: str
    answer_value: str
    answer_aliases: tuple[str, ...]
    answer_category: str
    category: str
    fact_comp_type: str

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> BridgeQuestion:
        """Build a question from one TwoHopFact CSV row."""
        fields = {name: str(row[column]) for name, column in _COLUMNS.items()}
        return cls(
            bridge_aliases=parse_aliases(row["e2.aliases"], fields["bridge_value"]),
            answer_aliases=parse_aliases(row["e3.aliases"], fields["answer_value"]),
            **fields,
        )

    @classmethod
    def from_dict(cls, record: Mapping) -> BridgeQuestion:
        """Rebuild a question from its `asdict()` form, as stored in JSONL."""
        return cls(
            **{
                **record,
                "bridge_aliases": tuple(record["bridge_aliases"]),
                "answer_aliases": tuple(record["answer_aliases"]),
            }
        )

    def as_dict(self) -> dict:
        return asdict(self)


def parse_aliases(raw: str, value: str) -> tuple[str, ...]:
    """Wikidata aliases as TwoHopFact stores them: the repr of a 1-tuple of a
    tuple of names.

    `value` heads the result, so a row whose alias cell is empty or
    unparseable still matches the entity under its canonical name rather than
    matching nothing at all.

    :param raw: the cell's text, e.g. ``"(('George Orwell', 'Eric Blair'),)"``
    :param value: the entity's canonical name
    :return: the canonical name followed by its aliases, deduplicated
    """
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        parsed = ()
    if isinstance(parsed, tuple) and parsed and isinstance(parsed[0], tuple):
        parsed = parsed[0]
    names = [name for name in parsed if isinstance(name, str)]
    return tuple(dict.fromkeys([value, *names]))


def read_questions(csv_path: Path) -> list[BridgeQuestion]:
    """Read every question in the TwoHopFact CSV.

    :param csv_path: the dataset CSV (see `twohopfact_csv`)
    :return: one question per row, in file order
    """
    # Alias and chain-of-thought cells run long; the module default would
    # reject them.
    csv.field_size_limit(1 << 20)
    with open(csv_path, newline="") as handle:
        return [BridgeQuestion.from_row(row) for row in csv.DictReader(handle)]


def twohopfact_csv(local_path: Path | None = None) -> Path:
    """The dataset CSV: a local copy if given, else the Hub's, downloaded once.

    :param local_path: an already-downloaded CSV to read instead
    :return: path to the CSV
    """
    if local_path is not None:
        return local_path
    from huggingface_hub import hf_hub_download

    from config import TWOHOPFACT_FILE, TWOHOPFACT_REPO

    return Path(
        hf_hub_download(
            repo_id=TWOHOPFACT_REPO, filename=TWOHOPFACT_FILE, repo_type="dataset"
        )
    )


def read_question_file(path: Path) -> list[BridgeQuestion]:
    """Read a JSONL question set (`filter_bridge_questions.py`'s output)."""
    with open(path) as handle:
        return [
            BridgeQuestion.from_dict(json.loads(line))
            for line in handle
            if line.strip()
        ]


def write_question_file(path: Path, questions: Iterable[BridgeQuestion]) -> None:
    """Write a JSONL question set, in the order given."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for question in questions:
            handle.write(json.dumps(question.as_dict()) + "\n")


def _filter_messages(
    shot: tuple[str, str, str], category: str, statement: str
) -> list[dict[str, str]]:
    shot_category, shot_statement, shot_answer = shot
    return [
        {
            "role": "user",
            "content": BRIDGE_FILTER_PROMPT.format(
                category=shot_category, statement=shot_statement
            ),
        },
        {"role": "assistant", "content": shot_answer},
        {
            "role": "user",
            "content": BRIDGE_FILTER_PROMPT.format(
                category=_DISPLAY_CATEGORY.get(category, category), statement=statement
            ),
        },
    ]


def bridge_hop_messages(question: BridgeQuestion) -> list[dict[str, str]]:
    """Ask the first hop on its own: does the model know the bridge entity?"""
    return _filter_messages(
        BRIDGE_FILTER_SHOT_BRIDGE, question.bridge_category, question.bridge_statement
    )


def answer_hop_messages(question: BridgeQuestion) -> list[dict[str, str]]:
    """Ask the two-hop question: does the model answer it right, without a
    chain of thought? The 1-shot exemplar answers with a bare name, which is
    what holds the model to a single-token-ish reply."""
    return _filter_messages(
        BRIDGE_FILTER_SHOT_ANSWER, question.answer_category, question.statement
    )


def answer_matches(reply: str, aliases: Sequence[str]) -> bool:
    """Did the model's free-text reply name one of `aliases`?

    Substring matching in both directions, as in the reference filter: a reply
    may wrap the name ("The answer is Paris") or give a shorter form than the
    alias ("Paris" for "Paris, France"). Loose on purpose -- it decides only
    which questions enter the sweep, and keeping it identical to the reference
    keeps the question set comparable to the paper's.

    :param reply: the model's answer text
    :param aliases: accepted names for the entity
    :return: whether the reply names the entity
    """
    normalized_reply = _normalize(reply)
    if not normalized_reply:
        return False
    return any(
        normalized_reply == alias
        or alias in normalized_reply
        or normalized_reply in alias
        for alias in (_normalize(a) for a in aliases)
        if alias
    )


def _normalize(text: str) -> str:
    return text.lower().strip().rstrip(".,!?;:")


def shuffled(
    questions: Sequence[BridgeQuestion], seed: int
) -> Iterator[BridgeQuestion]:
    """The dataset in a seeded random order.

    The filter walks this order and stops once it has enough accepted
    questions, so the order must be reproducible: a resumed run has to
    continue the same walk, not start a new one.
    """
    import random

    order = list(questions)
    random.Random(seed).shuffle(order)
    return iter(order)
