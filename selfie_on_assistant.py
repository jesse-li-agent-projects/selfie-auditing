"""SelfIE interpretation of the *assistant's own* response tokens.

Sibling of run_pipeline.py, which reads hidden states from the user turn. Here
each (arm, word) first gets one greedily decoded assistant reply to the
elicitation prompt, and the sweep then interprets the hidden state at every
token of that reply. The hypothesis is that a taboo model commits to its
secret while it speaks, not while it reads, so the two scripts' outputs are
complementary rather than alternatives -- nothing here re-covers the user-turn
positions run_pipeline.py already swept.

    python selfie_on_assistant.py --words book,chair \
        --output-dir outputs/taboo_assistant_armB/ \
        --adapter-path outputs/phase0_armB/best.pt

Cells are keyed by "resp<i>", the 0-based index of the token *within the
reply*, so unlike run_pipeline.py's end-relative offsets they are not
comparable across arms: each arm says something different and of its own
length. `--response-max-new-tokens` bounds that length, and so bounds the
sweep: truncating the reply is exact rather than approximate, since a causal
model's hidden state at token i cannot depend on tokens after it.

A run resumes: rerunning the same command against the same output directory
re-reads the cells already on disk, reuses the replies recorded in the
metadata sidecar, and generates only what is missing. A crash can leave the
last JSONL line half-written; delete the incomplete trailing lines and rerun.
The sharding flags work as in run_pipeline.py (see its docstring), with
merge_results.py combining shards afterwards.
"""

import argparse
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

# Light imports: neither config.py nor run_pipeline.py's module level pulls in
# torch, so --help stays fast.
from config import Arm
from run_pipeline import cell_seed, parse_arms, parse_layers


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--words", required=True, help="Comma-separated secret words to sweep"
    )
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--model", default=None, help="Base model repo (default: 8B)")
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Local path to the SelfIE adapter checkpoint (default: the 8B one, "
        "fetched once to config.SELFIE_ADAPTER_PATH)",
    )
    parser.add_argument(
        "--lora-template",
        default=None,
        help="Taboo LoRA repo/path template containing {word} (default: the 8B repos)",
    )
    parser.add_argument(
        "--arms",
        type=parse_arms,
        default=None,
        help="Comma-separated arms to sweep (default: control,prompted,finetuned)",
    )
    parser.add_argument(
        "--layers",
        type=parse_layers,
        default="all",
        help="'all' or a comma-separated list of 0-indexed layers (default: all)",
    )
    parser.add_argument(
        "--response-max-new-tokens",
        type=int,
        default=20,
        help="Tokens of assistant reply to interpret; one cell per token per "
        "layer (default: 20)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Generations per cell for this shard (default: the config's own)",
    )
    parser.add_argument(
        "--sample-start",
        type=int,
        default=0,
        help="Index of this shard's first generation, for seeding and merging",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Rows per forward pass, pooled across an (arm, word)'s cells "
        "(default: the config's own)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

import json

import torch
from jaxtyping import Float
from torch import Tensor
from transformers import AutoConfig

from adapter_training.inference import load_adapter
from config import BASE_MODEL_8B, resolve_layers, sweep_config
from extract import build_prompt, cache_path, save_tensors
from interpret import generate_interpretations_batch
from model_loading import (
    arm_active,
    attach_taboo_loras,
    load_base_model,
    load_tokenizer,
    system_prompt_for,
)
from preflight import check_config, check_output_dir, check_tokenization_pins
from results_store import (
    KEY_FIELDS,
    append_cell,
    cell_key,
    metadata_path,
    read_cells,
    read_metadata,
    shard_cells_path,
    write_metadata,
)
from scoring import contains_secret, score_cell


def response_position_key(index: int) -> str:
    """Cell/tensor key for the index-th token of the assistant response.

    Distinct from `extract.position_key`'s "pos<offset>" on purpose: these
    index a reply the model wrote, not the fixed prompt it read.

    :param index: 0-based token index within the response
    :return: the key used in the cells file and the hidden-state cache
    """
    return f"resp{index}"


@dataclass
class AssistantResponse:
    """One arm's reply to the elicitation prompt, and where it sits.

    Recorded in the metadata sidecar rather than only in memory: the cell keys
    are indices into this reply, so a resumed run must interpret the same
    tokens the interrupted one did. `reveals_secret` is kept because a cell's
    hit rate means something quite different when the reply said the word out
    loud (as CONTROL's usually does) than when it did not.
    """

    text: str
    prompt_len: int
    response_ids: list[int]
    tokens: dict[str, str]
    reveals_secret: bool


@torch.no_grad()
def generate_assistant_response(
    model,
    tokenizer,
    user_prompt: str,
    system_prompt: str | None,
    secret_word: str,
    max_new_tokens: int,
    device: str,
) -> AssistantResponse:
    """Greedily decode one reply to the elicitation prompt.

    Greedy, not sampled: the reply defines this run's cell space, so a rerun
    that produced a different one would silently be measuring different
    tokens under the same keys.

    :param model: the model, already in the arm's LoRA state
    :param tokenizer: tokenizer used to format the prompt and decode the reply
    :param user_prompt: the raw (unformatted) elicitation prompt
    :param system_prompt: system prompt for this arm, or None
    :param secret_word: the word this arm is hiding, for `reveals_secret`
    :param max_new_tokens: how much of the reply to keep
    :param device: device to generate on
    :return: the reply, its token ids, and the prompt length it follows
    :raises ValueError: if the model emitted no tokens at all
    """
    prompt_ids = tokenize_prompt(tokenizer, user_prompt, system_prompt, device)
    outputs = model.generate(
        input_ids=prompt_ids,
        attention_mask=torch.ones_like(prompt_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        # The base model's generation config presets these for sampling;
        # leaving them set alongside do_sample=False only earns a warning.
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = prompt_ids.shape[1]
    response_ids = outputs[0, prompt_len:].tolist()
    if not response_ids:
        raise ValueError("the model produced an empty assistant response")
    text = tokenizer.decode(response_ids, skip_special_tokens=True)
    print(f"[response] {len(response_ids)} tokens: {text!r}")
    return AssistantResponse(
        text=text,
        prompt_len=prompt_len,
        response_ids=response_ids,
        tokens={
            response_position_key(i): tokenizer.decode([token_id])
            for i, token_id in enumerate(response_ids)
        },
        reveals_secret=contains_secret(text, secret_word),
    )


def tokenize_prompt(
    tokenizer, user_prompt: str, system_prompt: str | None, device: str
) -> Tensor:
    """Render and tokenize the elicitation prompt, batched to one row."""
    formatted = build_prompt(tokenizer, user_prompt, system_prompt)
    return tokenizer(
        formatted, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)


@torch.no_grad()
def extract_response_hidden_states(
    model,
    prompt_ids: Tensor,
    response_ids: list[int],
    layers: list[int],
) -> dict[tuple[int, str], Float[Tensor, "hidden"]]:
    """Read every layer's hidden state at every token of the reply.

    A single teacher-forced pass over prompt + reply, not the states cached
    during generation: causally identical, and it keeps this independent of
    whether the reply was just generated or replayed from the metadata.
    Position `resp<i>` is the state the model held while reading its own token
    i -- the state that produced token i + 1.

    :param model: the model, in the same LoRA state the reply was written in
    :param prompt_ids: token ids of the formatted prompt, shape (1, prompt_len)
    :param response_ids: token ids of the reply
    :param layers: transformer layer indices to harvest
    :return: one hidden state per (layer, response position)
    """
    response = torch.tensor([response_ids], device=prompt_ids.device)
    input_ids = torch.cat([prompt_ids, response], dim=1)
    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        output_hidden_states=True,
    )
    prompt_len = prompt_ids.shape[1]
    return {
        (layer, response_position_key(i)): (
            # hidden_states[0] is the embedding output, so layer L is at L + 1.
            outputs.hidden_states[layer + 1][0, prompt_len + i, :]
            .float()
            .cpu()
        )
        for layer in layers
        for i in range(len(response_ids))
    }


def completed_keys(cells_path: Path) -> set[tuple]:
    """Which cells an earlier attempt at this shard already wrote.

    :param cells_path: this shard's cells file, which need not exist
    :return: the (arm, word, layer, position) keys already on disk
    :raises SystemExit: if a line is not valid JSON, i.e. a crash truncated it
    """
    if not cells_path.exists():
        return set()
    try:
        return {cell_key(cell) for cell in read_cells(cells_path)}
    except json.JSONDecodeError as error:
        raise SystemExit(
            f"{cells_path} has a malformed line ({error}) -- a crash left a "
            "partial write. Delete the trailing incomplete line(s) and rerun."
        )


def recorded_responses(cells_path: Path, metadata: dict) -> dict[str, dict[str, dict]]:
    """The replies an earlier attempt at this shard recorded, if it is a resume.

    Reusing them, rather than regenerating, is what makes the cell keys of a
    resumed run mean the same tokens as those already on disk. The settings
    check is deliberately strict: a directory whose sidecar disagrees with the
    run about weights, sampling or layers is not a resume of it.

    :param cells_path: this shard's cells file
    :param metadata: the metadata this run would write
    :return: recorded replies as arm -> word -> `AssistantResponse` fields
    :raises SystemExit: if the sidecar describes a different run
    """
    if not metadata_path(cells_path).exists():
        return {}
    previous = read_metadata(cells_path)
    differing = [
        field
        for field, value in metadata.items()
        if field != "responses" and previous.get(field) != value
    ]
    if differing:
        raise SystemExit(
            f"{metadata_path(cells_path)} was written by a run with different "
            f"{', '.join(differing)} -- resume it with its own settings, or "
            "point --output-dir at a fresh directory."
        )
    return previous.get("responses", {})


def run(
    config, *, adapter, tokenizer, peft_model, response_max_new_tokens, done
) -> Path:
    """Reply, extract, interpret and score, for every cell in `config`.

    :param config: this shard's pipeline config
    :param adapter: SelfIE adapter used to interpret each cell's hidden state
    :param tokenizer: tokenizer shared by generation and extraction
    :param peft_model: the (possibly LoRA-wrapped) model to speak with and extract from
    :param response_max_new_tokens: how much of each reply to interpret
    :param done: cells an earlier attempt already wrote, from `completed_keys`
    :return: path to this shard's cells file; its metadata sidecar sits beside it
    """
    sample_end = config.sample_start + config.n_samples
    cells_path = shard_cells_path(config.output_dir, config.sample_start, sample_end)
    responses: dict[str, dict[str, dict]] = {}

    def metadata() -> dict:
        return {
            "sample_range": [config.sample_start, sample_end],
            "batch_size": config.batch_size,
            "layers": config.layers,
            "response_max_new_tokens": response_max_new_tokens,
            "responses": responses,
            **config.comparable_settings(),
        }

    responses = recorded_responses(cells_path, metadata())

    # Written before the first cell so an interrupted shard is still
    # identifiable, and rewritten as each arm's reply becomes known.
    write_metadata(cells_path, metadata())
    with open(cells_path, "a") as handle:
        # arm: control/prompt/fine-tuned; word: which word is taboo
        for arm, word in product(config.arms, config.words):
            with arm_active(peft_model, arm, word):
                prompt_ids = tokenize_prompt(
                    tokenizer,
                    config.secret_prompt,
                    system_prompt_for(arm, word),
                    config.device,
                )
                recorded = responses.get(arm.value, {}).get(word)
                if recorded is None:
                    response = generate_assistant_response(
                        peft_model,
                        tokenizer,
                        config.secret_prompt,
                        system_prompt_for(arm, word),
                        word,
                        response_max_new_tokens,
                        config.device,
                    )
                else:
                    response = AssistantResponse(**recorded)
                    if response.prompt_len != prompt_ids.shape[1]:
                        raise SystemExit(
                            f"{arm.value}/{word}: the prompt is now "
                            f"{prompt_ids.shape[1]} tokens, but the recorded reply "
                            f"follows {response.prompt_len} -- the tokenizer or "
                            "chat template has changed since that run."
                        )
                responses.setdefault(arm.value, {})[word] = asdict(response)
                write_metadata(cells_path, metadata())

                hidden_states = extract_response_hidden_states(
                    peft_model, prompt_ids, response.response_ids, config.layers
                )
                save_tensors(
                    cache_path(config.output_dir, arm, word),
                    {
                        f"layer_{layer}__{key}": state
                        for (layer, key), state in hidden_states.items()
                    },
                )

                # No contrastive (mean-subtracted) preprocessing on the hidden
                # states, matching run_pipeline.py -- see its own note.
                pending = {
                    (layer, key): state
                    for (layer, key), state in hidden_states.items()
                    if (arm.value, word, layer, key) not in done
                }
                if not pending:
                    continue
                # One seed, one pooled batch of forward passes for every cell in
                # this (arm, word) -- see run_pipeline.cell_seed. A resumed run
                # reseeds identically but draws a shorter stream, so replay
                # identity holds only for a group generated in one attempt.
                torch.manual_seed(cell_seed(arm.value, word, config.sample_start))
                for (layer, key), generations in generate_interpretations_batch(
                    peft_model,
                    tokenizer,
                    adapter,
                    pending,
                    config.n_samples,
                    config.max_new_tokens,
                    config.temperature,
                    config.device,
                    config.batch_size,
                ):
                    cell = score_cell(generations, word)
                    append_cell(
                        handle,
                        dict(
                            zip(KEY_FIELDS, (arm.value, word, layer, key)),
                            generations=cell.generations,
                            hits=cell.hits,
                            hit_rate=cell.hit_rate,
                        ),
                    )
    return cells_path


def preflight(config, tokenizer, num_hidden_layers: int, response_max_new_tokens: int):
    """The checks of preflight.py that apply to a reply-token sweep.

    `check_run_prompts` is left out: it validates the user-prompt span, which
    this script does not extract from. The pinned tokenization still runs, so
    a chat-template change is still caught before any weights load.

    :raises preflight.PreflightError: on the first failed check
    """
    check_config(config, num_hidden_layers)
    check_output_dir(config)
    check_tokenization_pins(tokenizer)
    groups = len(config.arms) * len(config.words)
    print(
        f"[preflight] ok: {groups} (arm, word) groups x {len(config.layers)} layers "
        f"x up to {response_max_new_tokens} response tokens x "
        f"{config.n_samples} samples"
    )


def main(args) -> Path:
    """Load, sweep, write.

    :param args: parsed command-line arguments
    :return: path to this run's cells file; its metadata sidecar sits beside it
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model or BASE_MODEL_8B
    num_hidden_layers = AutoConfig.from_pretrained(model_name).num_hidden_layers
    kwargs = {
        name: value
        for name, value in (
            ("base_model", args.model),
            ("adapter_path", args.adapter_path),
            ("taboo_lora_repo_template", args.lora_template),
            ("n_samples", args.n_samples),
            ("batch_size", args.batch_size),
            ("max_new_tokens", args.max_new_tokens),
            ("temperature", args.temperature),
        )
        if value is not None
    }
    # No positions: this sweep's cells come from the reply, which does not
    # exist until the model has written it.
    config = sweep_config(
        args.words.split(","),
        layers=resolve_layers(args.layers, num_hidden_layers),
        arms=args.arms,
        positions=[],
        output_dir=output_dir,
        sample_start=args.sample_start,
        device=args.device,
        **kwargs,
    )
    tokenizer = load_tokenizer(config.base_model)
    preflight(config, tokenizer, num_hidden_layers, args.response_max_new_tokens)
    # Read before any weights load, so a resume of a corrupt file fails as fast
    # as any other precondition does.
    done = completed_keys(
        shard_cells_path(
            output_dir, config.sample_start, config.sample_start + config.n_samples
        )
    )
    if done:
        print(f"[resume] {len(done)} cells already written; generating the rest")
    model = load_base_model(config.base_model, device=args.device, dtype=args.dtype)
    adapter = load_adapter(config.adapter_path, device=args.device)

    peft_model = (
        attach_taboo_loras(model, config.words, config.taboo_lora_repo_template)
        if Arm.FINETUNED in config.arms
        else model
    )

    return run(
        config,
        adapter=adapter,
        tokenizer=tokenizer,
        peft_model=peft_model,
        response_max_new_tokens=args.response_max_new_tokens,
        done=done,
    )


if __name__ == "__main__":
    cells_path = main(args)
    print(f"Wrote {cells_path}, metadata in {metadata_path(cells_path)}")
