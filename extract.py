"""Forward pass and hidden-state extraction/caching (plan S3 step 1, S4.4).

One forward pass gives every layer's and every position's hidden state for
free; this module is only responsible for running that pass and caching the
handful of (layer, position) slices the sweep actually wants. Generation
(the expensive part) lives in interpret.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from jaxtyping import Float, Int
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file
from torch import Tensor

from config import Arm, Position


class TokenizerLike(Protocol):
    """The subset of the HF tokenizer interface this module depends on --
    lets find_positions() and user_prompt_span() be unit-tested without
    loading a real tokenizer."""

    def apply_chat_template(self, messages: list[dict], **kwargs) -> str: ...
    def __call__(self, text: str, **kwargs): ...
    def convert_tokens_to_ids(self, token: str) -> int: ...
    def decode(self, token_ids, **kwargs) -> str: ...


def build_prompt(
    tokenizer: TokenizerLike, user_prompt: str, system_prompt: str | None
) -> str:
    """Render the elicitation prompt (plan S4.5) with the chat template.

    `add_generation_prompt=True` puts the last token at the boundary right
    before the assistant would start speaking (research_notes S1.1) -- this is
    what makes the ASSISTANT_BOUNDARY position meaningful.
    """
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def find_positions(
    tokenizer: TokenizerLike, input_ids: Int[Tensor, "seq"]
) -> dict[Position, int]:
    """Locate the two candidate token positions in a formatted prompt (S4.4).

    ASSISTANT_BOUNDARY is always the last token (guaranteed by
    add_generation_prompt=True). LAST_CONTENT_TOKEN uses the *last* <|eot_id|>
    rather than the first or a fixed offset: CONTROL/PROMPTED prepend a system
    turn (containing the secret word itself) closed by its own <|eot_id|>, so
    only the last one is guaranteed to be the user turn's closer.
    """
    ids = input_ids.tolist()
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eot_positions = [i for i, t in enumerate(ids) if t == eot_id]
    if not eot_positions:
        raise ValueError(
            "no <|eot_id|> token found in formatted prompt -- chat template may "
            "not match the Llama-3 format this project assumes"
        )
    return {
        Position.ASSISTANT_BOUNDARY: len(ids) - 1,
        Position.LAST_CONTENT_TOKEN: eot_positions[-1] - 1,
    }


def user_prompt_span(
    tokenizer: TokenizerLike,
    input_ids: Int[Tensor, "seq"],
    user_prompt: str,
) -> list[int]:
    """Every token from the start of `user_prompt` to the end, as negative offsets.

    Because `build_prompt` renders with `add_generation_prompt=True`, the last
    token of the formatted prompt is the assistant boundary, so this span runs
    from the user prompt's first token through ASSISTANT_BOUNDARY inclusive --
    everything the model sees before it starts speaking, minus the system turn.

    Offsets are negative (end-relative) because absolute indices are not
    comparable across arms: a system turn shifts every absolute index, while
    the assistant boundary -- the alignment the arm comparison needs -- is
    always at -1.

    The span is located by decoding the prompt's own tokens and matching the
    text, anchoring on no named position and on no standalone tokenization of
    `user_prompt`. Both of those would bake in an assumption about the current
    chat template that a template change could break silently. The largest
    start whose slice still contains `user_prompt` gives the minimal span,
    which tolerates a first token that merges preceding template whitespace
    into the first content word, and takes the user turn's copy of the text
    even when an arm's system prompt quotes it verbatim.

    :param tokenizer: tokenizer used to decode candidate slices of `input_ids`
    :param input_ids: token ids of the fully formatted prompt
    :param user_prompt: the raw (unformatted) user prompt text to locate
    :return: negative, end-relative offsets covering every token of the span
    :raises ValueError: if no suffix of `input_ids` decodes to contain `user_prompt`
    """
    ids = input_ids.tolist()
    n = len(ids)
    for start in range(n - 1, -1, -1):
        if user_prompt in tokenizer.decode(ids[start:]):
            return list(range(start - n, 0))
    raise ValueError(
        f"no suffix of the formatted prompt contains {user_prompt!r} -- the "
        "chat template may have altered the prompt text"
    )


def expand_positions(
    tokenizer: TokenizerLike,
    input_ids: Int[Tensor, "seq"],
    user_prompt: str,
    positions: list[Position | int],
) -> list[Position | int]:
    """Replace the USER_PROMPT_SPAN sentinel with the offsets it stands for.

    Expansion happens here rather than in config because the offsets only
    exist once there are token ids to search. Duplicates are dropped by the
    token each position resolves to, so listing the sentinel alongside a named
    position inside the span does not produce two cells for one token.

    :param tokenizer: tokenizer used to locate the span and named positions
    :param input_ids: token ids of the fully formatted prompt
    :param user_prompt: the raw (unformatted) user prompt text
    :param positions: positions to expand; entries other than USER_PROMPT_SPAN pass through
    :return: `positions` with USER_PROMPT_SPAN replaced by its offsets, deduplicated by token
    """
    expanded: list[Position | int] = []
    for position in positions:
        if position is Position.USER_PROMPT_SPAN:
            expanded.extend(user_prompt_span(tokenizer, input_ids, user_prompt))
        else:
            expanded.append(position)

    pos_index = (
        find_positions(tokenizer, input_ids)
        if any(isinstance(p, Position) for p in expanded)
        else {}
    )
    n = len(input_ids)
    deduped: list[Position | int] = []
    seen: set[int] = set()
    for position in expanded:
        index = resolve_position(position, pos_index) % n
        if index not in seen:
            seen.add(index)
            deduped.append(position)
    return deduped


def resolve_position(position: Position | int, pos_index: dict[Position, int]) -> int:
    """Token index for a named position, or a raw index passed through as-is.

    A raw index addresses any token in the formatted prompt, including ones no
    named position covers. Negative values index from the end.
    """
    return pos_index[position] if isinstance(position, Position) else position


def position_key(position: Position | int) -> str:
    """Stable string key for a position, for tensor names and results files.

    :param position: a named position, or a raw token offset
    :return: the position's enum value, or "pos" followed by the raw offset
    """
    return position.value if isinstance(position, Position) else f"pos{position}"


@dataclass
class Extraction:
    """One forward pass's harvested cells, plus what they were read from.

    `tokens` records the decoded token each position actually resolved to. A
    stored offset like -11 only means a particular word relative to a
    particular formatted prompt, so carrying the decoding alongside the
    hidden states keeps results interpretable after a prompt or template
    change -- and makes two runs' comparability checkable instead of assumed.
    """

    hidden_states: dict[tuple[int, Position | int], Float[Tensor, "hidden"]]
    positions: list[Position | int]
    tokens: dict[str, str]


@torch.no_grad()
def extract_hidden_states(
    model,
    tokenizer: TokenizerLike,
    user_prompt: str,
    system_prompt: str | None,
    layers: list[int],
    positions: list[Position | int],
    device: str,
    verbose: bool = True,
) -> Extraction:
    """Run one forward pass and slice out every requested (layer, position) cell.

    hidden_states[0] is the embedding output; hidden_states[L + 1] is the
    output of transformer layer L (research_notes S1.1). `positions` may
    contain the USER_PROMPT_SPAN sentinel, which is expanded here.

    :param model: the model to run the forward pass on
    :param tokenizer: tokenizer used to format and index the prompt
    :param user_prompt: the raw (unformatted) user prompt text
    :param system_prompt: system prompt for this arm, or None
    :param layers: transformer layer indices to harvest
    :param positions: token positions to harvest, possibly including USER_PROMPT_SPAN
    :param device: device to run the forward pass on
    :param verbose: print each position's resolved token -- worth a line per
        position once per prompt shape, but not worth repeating for every one
        of a sweep's hundreds of prompts
    :return: the harvested hidden states, the expanded positions, and their decoded tokens
    """
    formatted = build_prompt(tokenizer, user_prompt, system_prompt)
    tokens = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(
        device
    )
    input_ids = tokens.input_ids[0]
    positions = expand_positions(tokenizer, input_ids, user_prompt, positions)
    pos_index = find_positions(tokenizer, input_ids)
    # Template drift across model/tokenizer versions is the one failure mode
    # this pipeline's outputs can't reveal on their own (plan S2's "silent
    # mismatch" concern generalizes here) -- print what got selected every
    # call. Cheap: one call per (arm, word), not per generation.
    decoded: dict[str, str] = {}
    for position in positions:
        idx = resolve_position(position, pos_index)
        key = position_key(position)
        decoded[key] = tokenizer.decode([input_ids[idx].item()])
        if verbose:
            print(f"[extract] {key} -> token {idx}: {decoded[key]!r}")

    outputs = model(
        input_ids=tokens.input_ids,
        attention_mask=tokens.attention_mask,
        output_hidden_states=True,
    )

    hidden_states = {}
    for layer in layers:
        for position in positions:
            idx = resolve_position(position, pos_index)
            hidden_states[(layer, position)] = (
                outputs.hidden_states[layer + 1][0, idx, :].float().cpu()
            )
    return Extraction(hidden_states=hidden_states, positions=positions, tokens=decoded)


def cache_path(output_dir: Path, arm: Arm, word: str) -> Path:
    """One cache file per (arm, word); each holds every swept (layer, position)."""
    return output_dir / "hidden_states" / arm.value / f"{word}.safetensors"


def _tensor_key(layer: int, position: Position | int) -> str:
    return f"layer_{layer}__{position_key(position)}"


def save_hidden_states(
    path: Path, hidden_states: dict[tuple[int, Position | int], Float[Tensor, "hidden"]]
) -> None:
    """Write the cache atomically.

    :param path: destination cache file
    :param hidden_states: cells to write, keyed by (layer, position)
    """
    save_tensors(
        path, {_tensor_key(layer, pos): t for (layer, pos), t in hidden_states.items()}
    )


def save_tensors(path: Path, tensors: dict[str, Float[Tensor, "hidden"]]) -> None:
    """Write a safetensors file atomically, creating its directory.

    Every shard of a sharded run recomputes byte-identical content and writes
    the same path, so concurrent writes are only benign if a reader can never
    observe a half-written file.

    :param path: destination file
    :param tensors: what to write, keyed by tensor name
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    safetensors_save_file(tensors, str(tmp_path))
    # safetensors always creates the file at mode 0600 regardless of umask,
    # which defeats a directory's group ACL (unlike a plain open()). Loosen it
    # back to the ACL's intent so the cache is readable by whoever can read
    # the rest of a run's output.
    tmp_path.chmod(0o664)
    os.replace(tmp_path, path)


def load_hidden_states(
    path: Path, layers: list[int], positions: list[Position | int]
) -> dict[tuple[int, Position | int], Float[Tensor, "hidden"]]:
    tensors = safetensors_load_file(str(path))
    return {
        (layer, position): tensors[_tensor_key(layer, position)]
        for layer in layers
        for position in positions
    }
