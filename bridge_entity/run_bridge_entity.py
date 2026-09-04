"""SelfIE bridge-entity detection over TwoHopFact (SelfIE adapter paper S3.6).

    python -m bridge_entity.filter_bridge_questions --output-dir bridge_entity
    python -m bridge_entity.run_bridge_entity --output-dir bridge_entity/bg_think \
        --adapter-path outputs/adapters/bg_think/best.pt
    python -m bridge_entity.run_bridge_entity --output-dir bridge_entity/baseline \
        --adapter-path outputs/adapters/wikipedia-scalar-affine.safetensors
    python -m bridge_entity.report_bridge_entity --run baseline=bridge_entity/baseline \
        --run bg_think=bridge_entity/bg_think

Reads a hidden state at every layer and every token of a two-hop question,
interprets each one through the adapter, and scores the interpretations for
any alias of the *bridge entity* -- the intermediate answer that appears in
neither the question nor the model's own reply. This is the paper's headline
out-of-distribution result: the adapter is applied unchanged, to raw
activations from a prompt shape it never trained on, with no mean
subtraction.

`--adapter-path untrained` swaps the checkpoint for the paper's untrained
SelfIE comparator (scale-only, no learned parameters).

Cells are appended to `cells.jsonl` as they finish, and every cell already in
that file is skipped on restart, so a machine crash costs at most the cells
in flight. A crash mid-write can leave a truncated final line; delete the
line and rerun. The run's settings are recorded beside the cells and checked
on restart, so a resumed run cannot silently mix two adapters into one file.
"""

import argparse
import hashlib
import json
from pathlib import Path

# Light import: config.py pulls in no heavy dependencies, so --help stays fast.
from config import BASE_MODEL_8B, outputs_relative, parse_layers

# What a resumed run must agree with its own earlier half about: which weights
# produced the generations, and what was sampled from them.
COMPARABLE_FIELDS = (
    "model",
    "adapter",
    "questions",
    "n_samples",
    "max_new_tokens",
    "temperature",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=outputs_relative,
        required=True,
        help="written under outputs/ (implicitly prepended)",
    )
    parser.add_argument(
        "--questions",
        type=outputs_relative,
        default="bridge_entity/questions.jsonl",
        help="filter_bridge_questions.py's output, under outputs/ "
        "(implicitly prepended)",
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="'untrained', a local .pt/.safetensors path, or a 'repo_id:filename' "
        "Hub pair",
    )
    parser.add_argument(
        "--untrained-scale",
        type=float,
        default=1.0,
        help="scale of the untrained comparator (default: upstream's "
        "identity_baseline.yaml; the reference bridge config used 10.0)",
    )
    parser.add_argument("--model", default=BASE_MODEL_8B)
    parser.add_argument(
        "--layers",
        type=parse_layers,
        default="all",
        help="'all' or a comma-separated list of 0-indexed layers (default: all)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="skip this many questions from the start of the set, e.g. to "
        "continue a preliminary run over a larger one (default: 0)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="use only the first N questions after --start-index (default: "
        "all of them)",
    )
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="rows per forward pass, pooled across a question's cells",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from adapter_training.checkpoints import (  # noqa: E402
    load_projection,
    untrained_projection,
)
from adapter_training.retrieval_eval import _ProjectionAdapter  # noqa: E402
from bridge_entity.bridge_dataset import (  # noqa: E402
    BridgeQuestion,
    read_question_file,
)
from extract import (  # noqa: E402
    build_prompt,
    extract_hidden_states,
    position_key,
    user_prompt_span,
)
from interpret import generate_interpretations_batch  # noqa: E402
from model_loading import load_base_model, load_tokenizer  # noqa: E402
from prompts import BRIDGE_STATEMENT_PROMPT  # noqa: E402
from results_store import (  # noqa: E402
    CELLS_FILE,
    append_cell,
    metadata_path,
    read_cells,
    write_metadata,
)
from scoring import contains_alias  # noqa: E402


def question_seed(question_id: str) -> int:
    """Deterministic per-question seed for the generation stream.

    blake2b rather than hash(): Python's hash() is salted per process, so a
    hash()-derived seed would give a different stream on every run. Per
    question rather than per cell because one question's cells share pooled
    forward passes (see `interpret.generate_interpretations_batch`), so only
    one RNG stream is ever live. A resumed run regenerates only the cells it
    is missing, which changes how that stream is consumed -- the draws are
    i.i.d. either way, but an exact replay needs an uninterrupted run.

    :param question_id: the question's TwoHopFact uid
    :return: a seed in [0, 2**31)
    """
    digest = hashlib.blake2b(question_id.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31)


def statement_positions(tokenizer, user_prompt: str, statement: str) -> list[int]:
    """Every token from the statement's first token to the assistant boundary,
    as end-relative offsets.

    Anchored on the statement rather than on the whole user prompt, so the
    span begins where the reference eval's fixed `start_token_position` does:
    past the "Complete the statement:" instruction, at the question's own
    first token. End-relative offsets because questions differ in length,
    and it is the boundary -- not the prompt's start -- that they align on.

    :param tokenizer: tokenizer used to format and index the prompt
    :param user_prompt: the full user turn, instruction included
    :param statement: the two-hop statement inside it
    :return: negative token offsets, in prompt order
    """
    formatted = build_prompt(tokenizer, user_prompt, None)
    input_ids = tokenizer(
        formatted, return_tensors="pt", add_special_tokens=False
    ).input_ids[0]
    return user_prompt_span(tokenizer, input_ids, statement)


def completed_cells(cells_path: Path) -> set[tuple[str, int, str]]:
    """The (question, layer, position) cells an earlier run already wrote.

    :param cells_path: this run's cells file
    :return: every cell key present, empty if the file does not exist
    :raises ValueError: if the file ends in a truncated line, which a crash
        mid-write can leave behind
    """
    if not cells_path.exists():
        return set()
    try:
        return {
            (cell["question_id"], cell["layer"], cell["position"])
            for cell in read_cells(cells_path)
        }
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{cells_path} ends in an incomplete line, most likely a crash "
            "mid-write -- delete the trailing line(s) and rerun"
        ) from error


def check_settings(cells_path: Path, settings: dict) -> None:
    """Record this run's settings, or check them against a resumed run's.

    :param cells_path: this run's cells file
    :param settings: the run's full settings, `COMPARABLE_FIELDS` included
    :raises ValueError: if a resumed run disagrees on any comparable field
    """
    if metadata_path(cells_path).exists():
        recorded = json.loads(metadata_path(cells_path).read_text())
        differing = {
            field: (recorded.get(field), settings[field])
            for field in COMPARABLE_FIELDS
            if recorded.get(field) != settings[field]
        }
        if differing:
            raise ValueError(
                f"{cells_path} was written with different settings {differing!r} -- "
                "write this run to its own output directory instead"
            )
    write_metadata(cells_path, settings)


def run(args, *, model, tokenizer, adapter, questions, layers) -> Path:
    """Sweep every (question, layer, token) cell, skipping ones already written.

    :param questions: the question set to sweep
    :param layers: transformer layer indices to read hidden states from
    :return: path to the cells file
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells_path = args.output_dir / CELLS_FILE
    done = completed_cells(cells_path)
    check_settings(
        cells_path,
        {
            "model": args.model,
            "adapter": args.adapter_path,
            "untrained_scale": args.untrained_scale,
            "questions": str(args.questions),
            "n_questions": len(questions),
            "layers": layers,
            "n_samples": args.n_samples,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "batch_size": args.batch_size,
            "statement_prompt": BRIDGE_STATEMENT_PROMPT,
        },
    )
    if done:
        print(f"Resuming: {len(done)} cells already in {cells_path}")

    progress = tqdm(questions, desc="questions")
    with open(cells_path, "a") as handle:
        for index, question in enumerate(progress):
            user_prompt = BRIDGE_STATEMENT_PROMPT.format(statement=question.statement)
            positions = statement_positions(tokenizer, user_prompt, question.statement)
            wanted = [
                (layer, position)
                for layer in layers
                for position in positions
                if (question.id, layer, position_key(position)) not in done
            ]
            if not wanted:
                continue

            # The template-drift check the printout exists for is a property
            # of the prompt shape, not of the question, so one question's
            # worth of it is enough for a 500-question sweep.
            extraction = extract_hidden_states(
                model,
                tokenizer,
                user_prompt,
                None,
                layers,
                positions,
                args.device,
                verbose=index == 0,
            )
            hidden = {key: extraction.hidden_states[key] for key in wanted}

            torch.manual_seed(question_seed(question.id))
            for (layer, position), generations in generate_interpretations_batch(
                model,
                tokenizer,
                adapter,
                hidden,
                args.n_samples,
                args.max_new_tokens,
                args.temperature,
                args.device,
                args.batch_size,
            ):
                hits = [
                    contains_alias(text, question.bridge_aliases)
                    for text in generations
                ]
                append_cell(
                    handle,
                    {
                        "question_id": question.id,
                        "layer": layer,
                        "position": position_key(position),
                        "token": extraction.tokens[position_key(position)],
                        "bridge_entity": question.bridge_value,
                        "generations": generations,
                        "hits": hits,
                        "hit_rate": sum(hits) / len(hits),
                    },
                )
    return cells_path


def load_questions(args) -> list[BridgeQuestion]:
    questions = read_question_file(args.questions)[args.start_index :]
    return questions[: args.max_questions] if args.max_questions else questions


def main(args) -> Path:
    from transformers import AutoConfig

    from config import resolve_layers

    model_config = AutoConfig.from_pretrained(args.model)
    layers = resolve_layers(args.layers, model_config.num_hidden_layers)
    questions = load_questions(args)
    print(f"Sweeping {len(questions)} questions over {len(layers)} layers")

    tokenizer = load_tokenizer(args.model)
    model = load_base_model(args.model, device=args.device, dtype=args.dtype)
    if args.adapter_path == "untrained":
        projection = untrained_projection(
            model_config.hidden_size,
            device=args.device,
            init_scale=args.untrained_scale,
        )
    else:
        projection, _ = load_projection(
            args.adapter_path, device=args.device, dim=model_config.hidden_size
        )
    adapter = _ProjectionAdapter(projection)

    return run(
        args,
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        questions=questions,
        layers=layers,
    )


if __name__ == "__main__":
    cells_path = main(args)
    print(f"Wrote {cells_path}, settings in {metadata_path(cells_path)}")
