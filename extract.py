"""Forward pass and hidden-state extraction/caching (plan S3 step 1, S4.4).

One forward pass gives every layer's and every position's hidden state for
free; this module is only responsible for running that pass and caching the
handful of (layer, position) slices the sweep actually wants. Generation
(the expensive part) lives in interpret.py.
"""

from __future__ import annotations

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
    lets find_positions() be unit-tested without loading a real tokenizer."""

    def apply_chat_template(self, messages: list[dict], **kwargs) -> str: ...
    def __call__(self, text: str, **kwargs): ...
    def convert_tokens_to_ids(self, token: str) -> int: ...


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

    ASSISTANT_BOUNDARY is always the last token (that's what
    add_generation_prompt=True guarantees). LAST_CONTENT_TOKEN is found by
    searching for the user turn's closing <|eot_id|> rather than hardcoding an
    offset from the end, so it survives chat-template changes across model
    versions.

    Uses the *last* <|eot_id|> in the sequence, not the first: CONTROL/PROMPTED
    render a system turn before the user turn, each closed by its own
    <|eot_id|>, and the first one belongs to the system turn -- which for this
    project's prompts contains the secret word itself. add_generation_prompt=True
    guarantees the sequence ends with an unclosed assistant header, so the last
    <|eot_id|> in the sequence is always the user turn's closer, regardless of
    how many turns (system or otherwise) precede it.
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


@torch.no_grad()
def extract_hidden_states(
    model,
    tokenizer: TokenizerLike,
    user_prompt: str,
    system_prompt: str | None,
    layers: list[int],
    positions: list[Position],
    device: str,
) -> dict[tuple[int, Position], Float[Tensor, "hidden"]]:
    """Run one forward pass and slice out every requested (layer, position) cell.

    hidden_states[0] is the embedding output; hidden_states[L + 1] is the
    output of transformer layer L (research_notes S1.1).
    """
    formatted = build_prompt(tokenizer, user_prompt, system_prompt)
    tokens = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(
        device
    )
    pos_index = find_positions(tokenizer, tokens.input_ids[0])
    # Template drift across model/tokenizer versions is the one failure mode
    # this pipeline's outputs can't reveal on their own (plan S2's "silent
    # mismatch" concern generalizes here) -- print what got selected every
    # call. Cheap: one call per (arm, word), not per generation.
    for position, idx in pos_index.items():
        token_str = tokenizer.decode([tokens.input_ids[0, idx].item()])
        print(f"[extract] {position.value} -> token {idx}: {token_str!r}")

    outputs = model(
        input_ids=tokens.input_ids,
        attention_mask=tokens.attention_mask,
        output_hidden_states=True,
    )

    result = {}
    for layer in layers:
        for position in positions:
            idx = pos_index[position]
            result[(layer, position)] = (
                outputs.hidden_states[layer + 1][0, idx, :].float().cpu()
            )
    return result


def cache_path(output_dir: Path, arm: Arm, word: str) -> Path:
    """One cache file per (arm, word); each holds every swept (layer, position)."""
    return output_dir / "hidden_states" / arm.value / f"{word}.safetensors"


def _tensor_key(layer: int, position: Position) -> str:
    return f"layer_{layer}__{position.value}"


def save_hidden_states(
    path: Path, hidden_states: dict[tuple[int, Position], Float[Tensor, "hidden"]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {_tensor_key(layer, pos): t for (layer, pos), t in hidden_states.items()}
    safetensors_save_file(tensors, str(path))
    # safetensors always creates the file at mode 0600 regardless of umask,
    # which defeats a directory's group ACL (unlike a plain open()). Loosen it
    # back to the ACL's intent so the cache is readable by whoever can read
    # the rest of a run's output.
    path.chmod(0o664)


def load_hidden_states(
    path: Path, layers: list[int], positions: list[Position]
) -> dict[tuple[int, Position], Float[Tensor, "hidden"]]:
    tensors = safetensors_load_file(str(path))
    return {
        (layer, position): tensors[_tensor_key(layer, position)]
        for layer in layers
        for position in positions
    }
