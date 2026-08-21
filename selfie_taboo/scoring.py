"""Secret-word hit-rate scoring and aggregation (plan S4.6).

Pure string logic, no heavy imports -- runs the same locally and on the remote.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Suffixes stripped/added to catch simple morphological variants (plan S4.6:
# "including simple plural/suffix variants"). Deliberately small and
# conservative -- the goal is to not miss "moons" or "danced", not to build a
# stemmer.
_SUFFIXES = ("s", "es", "ed", "ing")


def _word_variants(word: str) -> set[str]:
    variants = {word}
    variants.update(word + suffix for suffix in _SUFFIXES)
    if word.endswith("e"):
        variants.update(word[:-1] + suffix for suffix in ("ed", "ing"))
    return variants


def contains_secret(text: str, secret_word: str) -> bool:
    """Case-insensitive whole-word match for the secret word or a simple variant.

    Word-boundary matching avoids false positives from unrelated words that
    happen to contain the secret as a substring (e.g. "cat" inside "category").
    """
    variants = _word_variants(secret_word.lower())
    pattern = r"\b(" + "|".join(re.escape(v) for v in variants) + r")\b"
    return re.search(pattern, text.lower()) is not None


@dataclass
class CellResult:
    """Scored generations for one (arm x word x layer x position) cell."""

    generations: list[str]
    hits: list[bool]
    secret_word: str

    @property
    def n(self) -> int:
        return len(self.generations)

    @property
    def hit_rate(self) -> float:
        return sum(self.hits) / self.n if self.n else float("nan")


def score_cell(generations: list[str], secret_word: str) -> CellResult:
    """Score every generation in a cell for presence of its own secret word.

    Keeps every raw generation alongside its score (plan S4.6: "a 5% hit rate
    is uninterpretable without being able to read what the other 95% said").
    """
    hits = [contains_secret(g, secret_word) for g in generations]
    return CellResult(generations=generations, hits=hits, secret_word=secret_word)
