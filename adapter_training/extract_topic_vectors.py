"""Topic-vector extraction for the adapter experiment (plan S5.1, S5.2, step 1).

    python -m adapter_training.extract_topic_vectors --prompt-style pangram \
        --layer 19 --output-dir outputs/vectors/pangram_l19

Two prompt styles share one code path:

- `baseline` reproduces upstream extraction -- each topic's own conversational
  prompt, one vector read at the last prompt token.
- `pangram` asks the model to write one fixed sentence while thinking about the
  topic, and keeps one vector per sentence token.

The pangram style also filters. Rather than generating and string-comparing,
the same forward pass that reads the activations teacher-forces the sentence
and checks `argmax(logits[i - 1]) == forced[i]` at every response position: if
that holds everywhere, greedy decoding from the prompt would have produced
exactly the sentence and then stopped. So the filter verdict and the vectors
cost one forward pass together, and no decode loop is needed.

Vectors are written raw. The per-position means are written beside them and are
subtracted by the trainer, not here, so the centering choice can be revisited
without re-extracting (plan S5.3).
"""

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, cast

# Light import: config.py pulls in no heavy dependencies, so --help stays fast.
from config import BASE_MODEL_8B

DEFAULT_DATASET = "keenanpepper/fifty-thousand-things"
DEFAULT_DATASET_FILE = "wikipedia_vital_articles_level5_dataset.jsonl"

# The sentence the model is asked to write, and the response the filter demands
# back. The response carries a full stop that the instruction's quoted sentence
# does not; it is what the tokenizer's 10 pangram tokens (plan S4.2b) count, and
# `--response-text` exists so step 0 can measure the alternative without a code
# edit.
PANGRAM = "The quick brown fox jumps over the lazy dog"
DEFAULT_RESPONSE = PANGRAM + "."

# Verbatim from the user's request (plan S1). Do not reword: the whole
# experiment is a claim about this prompt.
PANGRAM_PROMPT_TEMPLATE = (
    'Write "{pangram}". Think about the topic "{topic}" while writing the '
    "sentence. Do not write anything else or change the words."
)


class PromptStyle(str, Enum):
    """Which extraction prompt a run reads activations from (plan S5.4)."""

    BASELINE = "baseline"  # upstream's per-topic prompt, one vector per topic
    PANGRAM = "pangram"  # the proposed prompt, one vector per response token


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Wikipedia topic vectors for adapter training."
    )
    parser.add_argument(
        "--prompt-style",
        type=PromptStyle,
        # The values, not the members: argparse renders `choices` verbatim, and
        # a str Enum still compares equal to its own value.
        choices=[style.value for style in PromptStyle],
        required=True,
    )
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=BASE_MODEL_8B)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--limit", type=int, default=None, help="only the first N topics"
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Hugging Face dataset id holding the topics",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=None,
        help="read topics from this local JSONL instead of the Hub -- for "
        "machines with no network egress",
    )
    parser.add_argument(
        "--response-text",
        default=DEFAULT_RESPONSE,
        help="the response the pangram filter demands, and whose tokens the "
        "vectors are read from",
    )
    return parser.parse_args()


# Parsed before the heavy imports below, so `--help` costs no torch import.
args = parse_args() if __name__ == "__main__" else None

import torch
from jaxtyping import Float, Int
from torch import Tensor

from extract import build_prompt


@dataclass(frozen=True)
class Topic:
    """One upstream dataset entry.

    `labels` are the natural-language descriptions the adapter is trained to
    emit; `split` is the dataset's own topic-level train/val assignment, which
    every vector of the topic inherits (plan S5.2).
    """

    title: str
    prompt: str
    labels: tuple[str, ...]
    split: str


@dataclass
class TopicRecord:
    """A surviving topic as written to `topics.json`.

    `start` and `count` address the topic's own vectors in `vectors.pt`, so
    labels are stored once per topic rather than once per vector.
    """

    title: str
    prompt: str
    labels: list[str]
    split: str
    start: int
    count: int


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


@dataclass
class ExtractionResult:
    """Everything one run writes, held in memory until the writers run."""

    vectors: Float[Tensor, "n_vectors hidden"]
    records: list[TopicRecord]
    position_tokens: list[str]
    position_means: Float[Tensor, "n_positions hidden"]
    failures: list[dict] = field(default_factory=list)
    n_seen: int = 0


def build_user_prompt(style: PromptStyle, topic: Topic, pangram: str = PANGRAM) -> str:
    """The unformatted user turn for one topic under one prompt style.

    The baseline style uses the topic's *own* hand-written prompt from the
    dataset (`Tell me about bits (binary digits).`, not a mechanical
    `Tell me about {title}.`) -- that is what upstream trained on. The pangram
    style takes only the title, since its prompt is fixed.

    :param style: which extraction prompt to build
    :param topic: the dataset entry
    :param pangram: the sentence the model is asked to write
    :return: the user turn, before the chat template is applied
    """
    if style is PromptStyle.BASELINE:
        return topic.prompt
    if style is PromptStyle.PANGRAM:
        return PANGRAM_PROMPT_TEMPLATE.format(pangram=pangram, topic=topic.title)
    raise ValueError(f"unknown prompt style: {style}")


def load_topics(
    dataset: str, dataset_file: Path | None = None, limit: int | None = None
) -> list[Topic]:
    """Read the upstream topic dataset, from the Hub or a local JSONL copy.

    :param dataset: Hugging Face dataset id, used when `dataset_file` is None
    :param dataset_file: local JSONL to read instead, for a machine with no egress
    :param limit: keep only the first N entries
    :return: topics in dataset order
    """
    rows: Iterable[dict[str, Any]]
    if dataset_file is not None:
        with open(dataset_file) as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        from datasets import load_dataset

        # `Dataset.__iter__` is typed as a union wide enough to include lists,
        # which it never yields for a JSONL-backed dataset.
        rows = cast(Iterable[dict[str, Any]], load_dataset(dataset, split="train"))

    topics = []
    for row in rows:
        topics.append(
            Topic(
                title=row["original_title"],
                prompt=row["prompt"],
                labels=tuple(row["labels"]),
                split=row["split"],
            )
        )
        if limit is not None and len(topics) == limit:
            break
    return topics


def response_token_ids(tokenizer, response_text: str) -> tuple[list[int], list[int]]:
    """Tokenize the forced assistant response.

    The trailing `<|eot_id|>` is forced as well as checked: without it a topic
    whose generation would have run on past the sentence -- adding commentary,
    say -- would pass the filter (plan S5.1).

    :param tokenizer: the model's tokenizer
    :param response_text: the sentence the model must reproduce
    :return: the sentence's token ids, and the full forced sequence (sentence + eot)
    """
    sentence_ids = tokenizer(response_text, add_special_tokens=False).input_ids
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    return sentence_ids, [*sentence_ids, eot_id]


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


def left_pad(
    sequences: list[list[int]], pad_id: int
) -> tuple[Int[Tensor, "batch seq"], Int[Tensor, "batch seq"]]:
    """Stack ragged token sequences with padding on the left.

    Left padding is what makes the response tokens land at fixed negative
    offsets for every example in the batch, so the caller can slice them
    without per-example bookkeeping.

    :param sequences: one token id list per example
    :param pad_id: the pad token id
    :return: the padded ids and their attention mask
    """
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        input_ids[row, width - len(sequence) :] = torch.tensor(
            sequence, dtype=torch.long
        )
        attention_mask[row, width - len(sequence) :] = 1
    return input_ids, attention_mask


def position_ids_from_mask(
    attention_mask: Int[Tensor, "batch seq"],
) -> Int[Tensor, "batch seq"]:
    """RoPE positions that ignore left padding.

    A plain forward pass defaults to `arange(seq_len)`, which under left
    padding shifts every real token's rotary position by that row's pad count
    -- so an example's activations would depend on which batch it landed in.
    Deriving positions from the mask keeps batched extraction bit-comparable
    with unbatched.

    :param attention_mask: 1 for real tokens, 0 for padding
    :return: position ids, zero where padded
    """
    return (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)


@torch.no_grad()
def extract_topics(
    model,
    tokenizer,
    topics: list[Topic],
    style: PromptStyle,
    layer: int,
    response_text: str = DEFAULT_RESPONSE,
    batch_size: int = 32,
    device: str = "cuda",
    pangram: str = PANGRAM,
    progress: bool = False,
) -> ExtractionResult:
    """Run the forward passes and harvest every kept vector.

    One vector per topic for the baseline style, one per response token for the
    pangram style. `hidden_states[L + 1]` is the output of transformer layer L.

    :param model: the base model to read activations from
    :param tokenizer: its tokenizer, configured for left padding
    :param topics: topics to extract, in the order they will be written
    :param style: which extraction prompt to use
    :param layer: transformer layer to read the residual stream at
    :param response_text: the response the pangram filter demands
    :param batch_size: topics per forward pass
    :param device: device to run on
    :param pangram: the sentence named in the pangram prompt
    :param progress: show a tqdm bar
    :return: vectors, the surviving topics, the per-position means and the filter failures
    """
    hidden_size = model.config.hidden_size
    if style is PromptStyle.PANGRAM:
        sentence_ids, forced_ids = response_token_ids(tokenizer, response_text)
        position_tokens = [tokenizer.decode([i]) for i in sentence_ids]
    else:
        sentence_ids, forced_ids = [], []
        position_tokens = ["last_prompt_token"]
    n_positions = len(position_tokens)

    vectors = torch.empty(
        len(topics) * n_positions, hidden_size, dtype=torch.bfloat16, device="cpu"
    )
    # Accumulated in float64 as the run goes rather than by casting the whole
    # 4 GB vector table at the end.
    sums = torch.zeros(n_positions, hidden_size, dtype=torch.float64)
    records: list[TopicRecord] = []
    failures: list[dict] = []
    written = 0

    batches = range(0, len(topics), batch_size)
    if progress:
        from tqdm import tqdm

        batches = tqdm(batches, desc=f"extracting ({style.value})")

    for start in batches:
        batch = topics[start : start + batch_size]
        sequences = [
            tokenizer(
                build_prompt(tokenizer, build_user_prompt(style, topic, pangram), None),
                add_special_tokens=False,
            ).input_ids
            + forced_ids
            for topic in batch
        ]
        input_ids, attention_mask = left_pad(sequences, tokenizer.pad_token_id)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids_from_mask(attention_mask),
            output_hidden_states=True,
            # The forced tokens sit at the last len(forced_ids) positions, and
            # each is predicted by the logits one position earlier -- so one
            # extra kept position covers the whole check. The baseline style
            # needs no logits at all, but 0 would mean "keep everything".
            logits_to_keep=len(forced_ids) + 1 if forced_ids else 1,
        )
        hidden = outputs.hidden_states[layer + 1]

        for row, topic in enumerate(batch):
            if style is PromptStyle.PANGRAM:
                compliance = check_forced_greedy(
                    outputs.logits[row, :-1, :].float(), forced_ids, tokenizer
                )
                if not compliance.ok:
                    failures.append({"title": topic.title, **asdict(compliance)})
                    continue
                # The forced block ends with <|eot_id|>, which is a response
                # token but carries no topic content, so it is checked and then
                # dropped.
                kept = hidden[row, -len(forced_ids) : -1, :]
            else:
                kept = hidden[row, -1:, :]

            kept = kept.to(dtype=torch.bfloat16, device="cpu")
            vectors[written : written + n_positions] = kept
            sums += kept.double()
            records.append(
                TopicRecord(
                    title=topic.title,
                    prompt=topic.prompt,
                    labels=list(topic.labels),
                    split=topic.split,
                    start=written,
                    count=n_positions,
                )
            )
            written += n_positions

    means = (sums / len(records)).float() if records else sums.float()
    return ExtractionResult(
        vectors=vectors[:written],
        records=records,
        position_tokens=position_tokens,
        position_means=means,
        failures=failures,
        n_seen=len(topics),
    )


def write_outputs(
    output_dir: Path,
    result: ExtractionResult,
    style: PromptStyle,
    layer: int,
    model_name: str,
    response_text: str,
) -> None:
    """Write the four artefacts the trainer and the report read (plan S5.2).

    `positions.json` is metadata only; the per-position means live beside it in
    `position_means.pt` because 10 x 4096 floats of JSON is half a megabyte of
    text to parse for no gain.

    :param output_dir: directory to create and write into
    :param result: what `extract_topics` returned
    :param style: the prompt style that produced it
    :param layer: the layer read
    :param model_name: the model read from, recorded for provenance
    :param response_text: the response the filter demanded
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.vectors, output_dir / "vectors.pt")
    torch.save(result.position_means, output_dir / "position_means.pt")

    with open(output_dir / "topics.json", "w") as handle:
        json.dump([asdict(record) for record in result.records], handle)

    n_labels = sum(len(record.labels) for record in result.records)
    with open(output_dir / "positions.json", "w") as handle:
        json.dump(
            {
                "prompt_style": style.value,
                "layer": layer,
                "model": model_name,
                "response_text": response_text if forced_style(style) else None,
                "n_positions": len(result.position_tokens),
                "position_tokens": result.position_tokens,
                "n_topics": len(result.records),
                "n_vectors": int(result.vectors.shape[0]),
                "hidden_size": int(result.vectors.shape[1]),
                "means_file": "position_means.pt",
            },
            handle,
            indent=2,
        )

    kept = len(result.records)
    with open(output_dir / "filter_report.json", "w") as handle:
        json.dump(
            {
                "topics_seen": result.n_seen,
                "topics_kept": kept,
                "topics_rejected": len(result.failures),
                "keep_rate": kept / result.n_seen if result.n_seen else 0.0,
                "labels_kept": n_labels,
                "train_topics": sum(r.split == "train" for r in result.records),
                "val_topics": sum(r.split == "val" for r in result.records),
                "first_mismatch_histogram": mismatch_histogram(result.failures),
                "failures": result.failures,
            },
            handle,
            indent=2,
        )


def forced_style(style: PromptStyle) -> bool:
    """Whether this prompt style teacher-forces a response, and so filters."""
    return style is PromptStyle.PANGRAM


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


def main(args) -> Path:
    from model_loading import load_base_model, load_tokenizer

    topics = load_topics(args.dataset, args.dataset_file, args.limit)
    print(f"Loaded {len(topics)} topics")

    tokenizer = load_tokenizer(args.model)
    model = load_base_model(args.model, device=args.device, dtype=args.dtype)

    result = extract_topics(
        model,
        tokenizer,
        topics,
        style=args.prompt_style,
        layer=args.layer,
        response_text=args.response_text,
        batch_size=args.batch_size,
        device=args.device,
        progress=True,
    )
    write_outputs(
        args.output_dir,
        result,
        args.prompt_style,
        args.layer,
        args.model,
        args.response_text,
    )
    print(
        f"Kept {len(result.records)}/{result.n_seen} topics, "
        f"{result.vectors.shape[0]} vectors"
    )
    return args.output_dir


if __name__ == "__main__":
    output_dir = main(args)
    print(
        f"Wrote {output_dir}/{{vectors,position_means}}.pt, topics.json, "
        f"positions.json, filter_report.json"
    )
