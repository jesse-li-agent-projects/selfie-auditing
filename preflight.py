"""Fail-fast checks that run before a sweep loads any weights.

    python preflight.py --words gold,moon --layers 32

Everything here is cheap -- config arithmetic, a tokenizer, a probe write --
and every check exists because the failure it catches would otherwise surface
minutes or hours into a run, after the expensive part had already started.

The pinned tokenization is the part no consistency check can replace. Comparing
arms to each other, or shards to each other, only proves they agree; a template
change that resolves to the same wrong token everywhere agrees with itself. Only
a recorded measurement catches that, so `PINNED_*` below is measured truth and is
meant to be updated deliberately, in a reviewable diff, when the prompt changes.
"""

from __future__ import annotations

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--words", required=True, help="Comma-separated secret words to check"
    )
    parser.add_argument(
        "--layers",
        type=int,
        required=True,
        help="The base model's layer count (from its config.json)",
    )
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


from config import SECRET_PROMPT, Arm, PipelineConfig, Position

# Measured against the real tokenizer for PIN_WORD (plan S2, S6.2). The 1B
# smoke tokenizer and the gated 8B agree, so these hold for both.
PIN_WORD = "gold"
PINNED_SPAN = list(range(-11, 0))
PINNED_SPAN_TOKENS = [
    "What",
    " is",
    " the",
    " secret",
    " word",
    "?",
    "<|eot_id|>",
    "<|start_header_id|>",
    "assistant",
    "<|end_header_id|>",
    "\n\n",
]
# Pinned as well as the span: the span offsets alone would not have caught the
# mirror-model discrepancy of plan S2, where every absolute index moved by
# 20-25 tokens while the span stayed at -11 .. -1.
PINNED_PROMPT_LENGTHS = {Arm.CONTROL: 62, Arm.PROMPTED: 66, Arm.FINETUNED: 41}


class PreflightError(Exception):
    """A precondition of the run is wrong. Raised before any weights load."""


def check_config(config: PipelineConfig, num_hidden_layers: int) -> None:
    """Check the config's own arithmetic against the model it will run on.

    :param config: the config this shard would run
    :param num_hidden_layers: the base model's layer count, from its own config.json
    :raises PreflightError: if a layer index, sample range, or word list is unusable
    """
    if not config.words:
        raise PreflightError("no words to sweep")
    if len(set(config.words)) != len(config.words):
        raise PreflightError(
            f"duplicate words in {config.words} -- cells would collide"
        )
    out_of_range = [
        layer for layer in config.layers if not 0 <= layer < num_hidden_layers
    ]
    if out_of_range:
        raise PreflightError(
            f"layers {out_of_range} are outside the model's {num_hidden_layers} "
            "layers -- the config's layer count disagrees with the model's own"
        )
    if config.n_samples < 1:
        raise PreflightError(f"n_samples must be at least 1, got {config.n_samples}")
    if config.sample_start < 0:
        raise PreflightError(
            f"sample_start must not be negative, got {config.sample_start}"
        )


def check_output_dir(config: PipelineConfig) -> None:
    """Check the output directory can actually be written to.

    A sweep writes nothing durable until its first cell finishes, so an
    unwritable directory would otherwise be discovered long after launch.

    :param config: the config this shard would run
    :raises PreflightError: if the output directory cannot be created or written
    """
    probe = config.output_dir / ".preflight"
    try:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        probe.write_text("")
        probe.unlink()
    except OSError as error:
        raise PreflightError(f"output dir {config.output_dir} is not writable: {error}")


def check_tokenization_pins(tokenizer) -> None:
    """Compare this tokenizer's rendering of the pinned prompt to the measurement.

    Deliberately independent of the run's own config: it asks the fixed
    question about PIN_WORD, so the answer is a property of the tokenizer and
    chat template alone and drifts only when one of those changes.

    :param tokenizer: the tokenizer the run will use
    :raises PreflightError: if the span, its tokens, or a prompt length has drifted
    """
    for arm in Arm:
        ids, span, tokens = _render(tokenizer, arm, PIN_WORD, SECRET_PROMPT)
        if len(ids) != PINNED_PROMPT_LENGTHS[arm]:
            raise _drift(arm, "prompt length", PINNED_PROMPT_LENGTHS[arm], len(ids))
        if span != PINNED_SPAN:
            raise _drift(arm, "span offsets", PINNED_SPAN, span)
        if tokens != PINNED_SPAN_TOKENS:
            raise _drift(arm, "span tokens", PINNED_SPAN_TOKENS, tokens)


def check_run_prompts(tokenizer, config: PipelineConfig) -> None:
    """Check every (arm, word) this run will actually sweep, structurally.

    The pins cover one prompt; this covers the run's own. It asserts the three
    properties the arm comparison depends on: the span is the same offsets
    everywhere (only end-relative offsets align arms whose system turns differ
    in length), it ends at ASSISTANT_BOUNDARY, and it holds the question but
    never the secret word -- a span that crept into the system turn would hand
    the interpreter the answer outright.

    :param tokenizer: the tokenizer the run will use
    :param config: the config this shard would run
    :raises PreflightError: if any (arm, word) resolves a different or wrong span
    """
    from extract import find_positions

    reference: tuple[list[int], list[str]] | None = None
    for arm in config.arms:
        for word in config.words:
            ids, span, tokens = _render(tokenizer, arm, word, config.secret_prompt)
            text = tokenizer.decode([ids[o] for o in span])
            if not text.startswith(config.secret_prompt):
                raise PreflightError(
                    f"{arm.value}/{word}: the span decodes to {text!r}, which does "
                    f"not start with the prompt {config.secret_prompt!r}"
                )
            # Past the question itself, the span is template tokens only, so
            # the secret word can only appear there if the span has slipped.
            tail = text[len(config.secret_prompt) :]
            if word.lower() in tail.lower():
                raise PreflightError(
                    f"{arm.value}/{word}: the secret word appears inside the span "
                    f"({text!r}) -- the span has reached into the system turn"
                )
            boundary = find_positions(tokenizer, ids)[Position.ASSISTANT_BOUNDARY]
            if len(ids) + span[-1] != boundary:
                raise PreflightError(
                    f"{arm.value}/{word}: the span ends at offset {span[-1]}, not at "
                    "ASSISTANT_BOUNDARY -- it does not cover the whole user turn"
                )
            if reference is None:
                reference = (span, tokens)
            elif (span, tokens) != reference:
                raise PreflightError(
                    f"{arm.value}/{word} resolves {span} -> {tokens}, but another "
                    f"cell resolves {reference[0]} -> {reference[1]} -- the arms "
                    "are not measuring the same token"
                )


def preflight(config: PipelineConfig, tokenizer, num_hidden_layers: int) -> None:
    """Run every check, then report what the run will read.

    :param config: the config this shard would run
    :param tokenizer: the tokenizer the run will use
    :param num_hidden_layers: the base model's layer count, from its own config.json
    :raises PreflightError: on the first failed check
    """
    check_config(config, num_hidden_layers)
    check_output_dir(config)
    check_tokenization_pins(tokenizer)
    check_run_prompts(tokenizer, config)
    cells = len(config.arms) * len(config.words) * len(config.layers) * len(PINNED_SPAN)
    print(
        f"[preflight] ok: span {PINNED_SPAN[0]} .. {PINNED_SPAN[-1]} = "
        f"{PINNED_SPAN_TOKENS}, {cells} cells x {config.n_samples} samples"
    )


def _render(tokenizer, arm: Arm, word: str, user_prompt: str):
    """Format one (arm, word) prompt and locate its span. Returns (ids, span, tokens)."""
    import torch

    from extract import build_prompt, user_prompt_span
    from model_loading import system_prompt_for

    formatted = build_prompt(tokenizer, user_prompt, system_prompt_for(arm, word))
    ids = tokenizer(formatted, add_special_tokens=False).input_ids
    ids = torch.tensor(ids)
    span = user_prompt_span(tokenizer, ids, user_prompt)
    return ids, span, [tokenizer.decode([ids[o]]) for o in span]


def _drift(arm: Arm, what: str, expected, got) -> PreflightError:
    return PreflightError(
        f"{arm.value}: {what} is {got!r}, but the pinned measurement is "
        f"{expected!r} -- the tokenizer or chat template has changed. If the "
        "change is intended, update the PINNED_* values in preflight.py."
    )


if __name__ == "__main__":
    args = parse_args()

    from pathlib import Path

    from config import full_sweep_config
    from model_loading import load_tokenizer

    config = full_sweep_config(
        args.words.split(","),
        num_hidden_layers=args.layers,
        output_dir=Path(args.output_dir),
    )
    preflight(config, load_tokenizer(config.base_model), args.layers)
