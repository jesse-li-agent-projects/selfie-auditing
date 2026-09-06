"""The forced-pangram compliance filter and the extraction loop built on it,
shared by `extract_pangram_vectors` (one topic per prompt) and
`extract_grouped_vectors` (one to three).

Rather than generating and string-comparing, the same forward pass that reads
the activations teacher-forces the sentence and checks
`argmax(logits[i - 1]) == forced[i]` at every response position: if that holds
everywhere, greedy decoding from the prompt would have produced exactly the
sentence and then stopped. So the filter verdict and the vectors cost one
forward pass together, and no decode loop is needed.

The sentence has two compliant shapes, with and without a trailing full stop,
and the filter accepts either (`response_variants`) -- so `count` is not
constant across items.

Nothing here knows how many topics the prompt named: the caller supplies the
instruction text and the record to write for each item.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

import torch
from jaxtyping import Float
from torch import Tensor

from adapter_training.dataset import GroupRecord, TopicRecord
from adapter_training.extract_common import (
    ExtractionResult,
    run_forward,
    write_extraction_outputs,
)
from extract import build_prompt

# The sentence the model is asked to write, and the response the filter
# demands back. The response carries a full stop that the instruction's
# quoted sentence does not; it is what the tokenizer's 10 pangram tokens
# count.
PANGRAM = "The quick brown fox jumps over the lazy dog"
DEFAULT_RESPONSE = PANGRAM + "."

Item = TypeVar("Item")
Record = TypeVar("Record", TopicRecord, GroupRecord)


@dataclass(frozen=True)
class Compliance:
    """Whether greedy decoding would have reproduced the forced response.

    A failure records where it first diverged, which is what turns the filter
    report into a failure taxonomy rather than a bare count.
    """

    ok: bool
    mismatch_index: int | None = None
    expected: str | None = None
    predicted: str | None = None


def response_token_ids(tokenizer, response_text: str) -> tuple[list[int], list[int]]:
    """Tokenize the forced assistant response.

    The trailing `<|eot_id|>` is forced as well as checked: without it an item
    whose generation would have run on past the sentence -- adding commentary,
    say -- would pass the filter.

    :param tokenizer: the model's tokenizer
    :param response_text: the sentence the model must reproduce
    :return: the sentence's token ids, and the full forced sequence (sentence + eot)
    """
    sentence_ids = tokenizer(response_text, add_special_tokens=False).input_ids
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    return sentence_ids, [*sentence_ids, eot_id]


def response_variants(tokenizer) -> list[tuple[str, list[int], list[int]]]:
    """The forced-sequence candidates the filter accepts for one item.

    The pangram has two compliant shapes: with the trailing full stop and
    without it, the model simply stopping one token earlier. Both are tried
    per item, longer first, and an item is kept on the first that matches --
    a filter forcing only one shape would structurally cap the keep rate near
    whichever fraction it picked.

    In practice the no-stop shape is now rare: moving the full stop inside the
    prompt's quotes (`dog."` rather than `dog".`) removed the ambiguity, and
    all 47,001 kept topics of the layer-19 run matched the with-stop shape.
    The fallback stays because it is cheap and because a new prompt (more
    background topics, say) is not covered by that measurement.

    The no-stop candidate is included only if stripping the stop leaves a
    genuine token-level prefix of the with-stop one (guards against a
    tokenizer merging the stop into the preceding word, which would make the
    two sequences unrelated rather than one a prefix of the other).

    :param tokenizer: the model's tokenizer
    :return: one or two `(text, sentence_ids, forced_ids)` candidates
    """
    sentence_ids, forced_ids = response_token_ids(tokenizer, DEFAULT_RESPONSE)
    no_stop_ids, no_stop_forced_ids = response_token_ids(tokenizer, PANGRAM)
    variants = [(DEFAULT_RESPONSE, sentence_ids, forced_ids)]
    if no_stop_ids and no_stop_ids == sentence_ids[: len(no_stop_ids)]:
        variants.append((PANGRAM, no_stop_ids, no_stop_forced_ids))
    return variants


def check_forced_greedy(
    logits: Float[Tensor, "n_forced vocab"], forced_ids: list[int], tokenizer
) -> Compliance:
    """Would greedy decoding have produced `forced_ids`?

    `logits[i]` must be the distribution over the token at `forced_ids[i]`, so
    the caller is responsible for the one-position shift. Agreement at every
    step *is* the greedy output: greedy decoding is deterministic and would
    have followed this same prefix.

    :param logits: next-token logits aligned to `forced_ids`
    :param forced_ids: the teacher-forced response tokens
    :param tokenizer: used only to decode a mismatch for the report
    :return: the verdict, plus the first divergence if there was one
    """
    predicted = logits.argmax(dim=-1).tolist()
    for index, (want, got) in enumerate(zip(forced_ids, predicted)):
        if want != got:
            return Compliance(
                ok=False,
                mismatch_index=index,
                expected=tokenizer.decode([want]),
                predicted=tokenizer.decode([got]),
            )
    return Compliance(ok=True)


def mismatch_histogram(failures: list[dict]) -> dict[str, int]:
    """Count rejections by the response position they first diverged at.

    Divergence at position 0 is a different failure from divergence at the
    final `<|eot_id|>`: the first means the model never started the sentence,
    the last means it started but did not stop.
    """
    histogram: dict[str, int] = {}
    for failure in failures:
        key = str(failure["mismatch_index"])
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items(), key=lambda item: int(item[0])))


@dataclass
class ForwardPass:
    """One variant's forward-pass results for a batch, alongside the
    candidate it forced -- what `select_variant` picks between."""

    text: str
    sentence_ids: list[int]
    forced_ids: list[int]
    logits: Float[Tensor, "batch n_forced vocab"]
    hidden: Float[Tensor, "batch seq hidden"]


def select_variant(
    passes: list[ForwardPass], row: int, tokenizer
) -> tuple[ForwardPass, None] | tuple[None, tuple[str, Compliance]]:
    """Pick the first variant whose greedy decoding matches, for one row.

    Tries each variant in order (longer, more common shape first) and keeps
    the first match. If none match, returns the variant whose greedy decoding
    got furthest, so a rejection points at the real divergence rather than an
    artefact of variant order.

    :param passes: one `ForwardPass` per variant, from the same batch
    :param row: which example in the batch to decide for
    :param tokenizer: used by `check_forced_greedy` to decode a mismatch
    :return: `(chosen_pass, None)` on a match, else `(None, (variant_text, compliance))`
    """
    worst_failure: tuple[str, Compliance] | None = None
    for forward_pass in passes:
        compliance = check_forced_greedy(
            forward_pass.logits[row, :-1, :].float(), forward_pass.forced_ids, tokenizer
        )
        if compliance.ok:
            return forward_pass, None
        # `or -1` would be wrong here: a genuine mismatch_index of 0 is falsy too.
        this_index = (
            compliance.mismatch_index if compliance.mismatch_index is not None else -1
        )
        prior_index = (
            worst_failure[1].mismatch_index
            if worst_failure is not None and worst_failure[1].mismatch_index is not None
            else -1
        )
        if worst_failure is None or this_index > prior_index:
            worst_failure = (forward_pass.text, compliance)
    assert worst_failure is not None
    return None, worst_failure


@torch.no_grad()
def extract_forced_vectors(
    model,
    tokenizer,
    items: Sequence[Item],
    *,
    layer: int,
    instruction_of: Callable[[Item], str],
    record_of: Callable[[Item, str, int, int], Record],
    identify: Callable[[Item], dict[str, Any]],
    batch_size: int = 32,
    device: str = "cuda",
    progress: bool = False,
    description: str = "extracting",
) -> ExtractionResult:
    """Run the forward passes and harvest every kept vector.

    One vector per response token of whichever forced variant (with or
    without the trailing full stop) the item's greedy decoding actually
    matches -- each variant gets its own forward pass per batch, tried in
    order, first match wins (`response_variants`, `select_variant`). An item
    that matches neither is rejected. `hidden_states[L + 1]` is the output of
    transformer layer L.

    :param model: the base model to read activations from
    :param tokenizer: its tokenizer, configured for left padding
    :param items: whatever a prompt is built from -- one topic, or one group
    :param layer: transformer layer to read the residual stream at
    :param instruction_of: the extraction prompt naming this item's topic(s)
    :param record_of: build this item's record, given `(item, variant_text,
        start, count)`
    :param identify: the fields naming a rejected item in the filter report
    :param batch_size: items per forward pass
    :param device: device to run on
    :param progress: show a tqdm bar
    :param description: the progress bar's label
    :return: vectors, the surviving records, the per-position means and the failures
    """
    hidden_size = model.config.hidden_size
    variants = response_variants(tokenizer)
    position_tokens = [tokenizer.decode([i]) for i in variants[0][1]]
    n_positions = len(position_tokens)

    vectors = torch.empty(
        len(items) * n_positions, hidden_size, dtype=torch.bfloat16, device="cpu"
    )
    # Accumulated in float64 as the run goes rather than by casting the whole
    # 4 GB vector table at the end. Counted per position, not just divided by
    # len(records), because a shorter (no-stop) variant leaves the last
    # position's count below the others.
    sums = torch.zeros(n_positions, hidden_size, dtype=torch.float64)
    position_counts = torch.zeros(n_positions, dtype=torch.float64)
    records: list[Record] = []
    failures: list[dict] = []
    written = 0

    batches = range(0, len(items), batch_size)
    if progress:
        from tqdm import tqdm

        batches = tqdm(batches, desc=description)

    for start in batches:
        batch = items[start : start + batch_size]
        prompts = [
            build_prompt(tokenizer, instruction_of(item), None) for item in batch
        ]
        passes = [
            ForwardPass(
                text,
                sentence_ids,
                forced_ids,
                *run_forward(model, tokenizer, prompts, forced_ids, layer, device),
            )
            for text, sentence_ids, forced_ids in variants
        ]

        for row, item in enumerate(batch):
            chosen, failure = select_variant(passes, row, tokenizer)
            if chosen is None:
                assert failure is not None
                text, compliance = failure
                failures.append(
                    {**identify(item), "variant": text, **asdict(compliance)}
                )
                continue

            # The forced block ends with <|eot_id|>, which is a response
            # token but carries no topic content, so it is checked and then
            # dropped.
            kept = chosen.hidden[row, -len(chosen.forced_ids) : -1, :]
            n_kept = kept.shape[0]
            kept = kept.to(dtype=torch.bfloat16, device="cpu")
            vectors[written : written + n_kept] = kept
            sums[:n_kept] += kept.double()
            position_counts[:n_kept] += 1
            records.append(record_of(item, chosen.text, written, n_kept))
            written += n_kept

    means = (sums / position_counts.clamp(min=1).unsqueeze(-1)).float()
    return ExtractionResult(
        vectors=vectors[:written],
        records=records,
        means=means,
        n_seen=len(items),
        position_tokens=position_tokens,
        failures=failures,
    )


def write_forced_outputs(
    output_dir: Path,
    result: ExtractionResult,
    *,
    layer: int,
    model_name: str,
    prompt_style: str,
    n_labels: int,
    unit: str = "topics",
    records_file: str = "topics.json",
    extra_positions: dict[str, Any] | None = None,
) -> None:
    """`write_extraction_outputs` plus `positions.json` and
    `filter_report.json`, the two artefacts the forced-pangram styles add.

    :param prompt_style: recorded in `positions.json`; how a reader tells a
        single-topic directory from a grouped one
    :param n_labels: total labels behind the kept records, counted by the
        caller because the label shape differs per style
    :param unit: what one record is, used to name the count keys of the filter
        report ("topics" or "groups")
    :param records_file: the file `write_extraction_outputs` writes records to
    :param extra_positions: style-specific `positions.json` fields
    """
    write_extraction_outputs(output_dir, result, records_file=records_file)

    assert result.position_tokens is not None
    with open(output_dir / "positions.json", "w") as handle:
        json.dump(
            {
                "prompt_style": prompt_style,
                "layer": layer,
                "model": model_name,
                "response_text": DEFAULT_RESPONSE,
                "n_positions": len(result.position_tokens),
                "position_tokens": result.position_tokens,
                f"n_{unit}": len(result.records),
                "n_vectors": int(result.vectors.shape[0]),
                "hidden_size": int(result.vectors.shape[1]),
                "means_file": "position_means.pt",
                **(extra_positions or {}),
            },
            handle,
            indent=2,
        )

    kept = len(result.records)
    variant_counts: dict[str, int] = {}
    for record in result.records:
        assert record.variant is not None
        variant_counts[record.variant] = variant_counts.get(record.variant, 0) + 1
    with open(output_dir / "filter_report.json", "w") as handle:
        json.dump(
            {
                f"{unit}_seen": result.n_seen,
                f"{unit}_kept": kept,
                f"{unit}_rejected": len(result.failures),
                "keep_rate": kept / result.n_seen if result.n_seen else 0.0,
                "labels_kept": n_labels,
                f"train_{unit}": sum(r.split == "train" for r in result.records),
                f"val_{unit}": sum(r.split == "val" for r in result.records),
                # Which forced variant each kept item matched, not just a
                # pass/fail count: a distribution far from the one
                # `response_variants` documents means an unusual sample.
                "variant_counts": variant_counts,
                "first_mismatch_histogram": mismatch_histogram(result.failures),
                "failures": result.failures,
            },
            handle,
            indent=2,
        )
