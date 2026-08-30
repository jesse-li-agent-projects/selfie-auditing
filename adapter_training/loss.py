"""Soft-prompt forward pass and cross-entropy loss, reproducing
`SelfIEModel.compute_loss` (resources/selfie-adapters/training/model.py)
exactly in what it *measures* while batching differently.

Two departures from upstream, both exact:

- **Logit slicing.** Upstream materialises logits for the whole padded
  sequence via the ordinary CausalLM forward, then loops in Python over the
  batch. Here the base transformer (no `lm_head`) is called once, its final
  hidden state is sliced down to the positions that predict target tokens,
  and `lm_head` runs on only that slice -- one batched `cross_entropy`
  instead of a per-example loop, ~75% less logits memory.
- **Right padding with an attention mask**, as upstream does. The injection
  slots are fixed offsets from the *left* of the template, so left padding
  would move them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from adapter_training.dataset import Example
from interpret import RESERVED_TOKEN, SELFIE_TEMPLATE


@dataclass(frozen=True)
class LossConfig:
    """Everything upstream's `compute_loss` takes as a loss parameter, plus
    target construction (upstream's `VectorLabelDataset.__getitem__`)."""

    max_loss: float = 100.0  # upstream TrainingConfig default; per-token clamp
    label_smoothing: float = 0.0
    strip_labels: bool = True
    eos_token: str = "<|eot_id|>"


def target_text(label: str, config: LossConfig) -> str:
    """The exact target string upstream trains against
    (`VectorLabelDataset.__getitem__`): `label + '"' + eos`.

    :param label: a topic's label, as stored in `topics.json`
    :param config: supplies `strip_labels` and `eos_token`
    """
    label = label.strip() if config.strip_labels else label
    return label + '"' + config.eos_token


class SoftPromptLoss:
    """Injects a vector into the fixed SelfIE interpretation template and
    scores the cross-entropy of the model completing it with a label.

    Constructed once per (model, projection); `__call__` is the per-batch
    forward + loss. `model` must be a HF-style CausalLM exposing `.model`
    (the base transformer, sans `lm_head`) and `.lm_head` -- true of every
    model this project uses (Llama).
    """

    def __init__(self, model, tokenizer, projection, config: LossConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.projection = projection
        self.config = config

        self.base_model = model.model
        self.lm_head = model.lm_head
        self.embed_layer = model.get_input_embeddings()

        template_ids = self.tokenizer(
            SELFIE_TEMPLATE, add_special_tokens=False
        ).input_ids
        reserved_id = self.tokenizer.convert_tokens_to_ids(RESERVED_TOKEN)
        # Scanned, not hardcoded: for the real template/tokenizer this lands
        # at [11, 22], but the invariant this module depends on is just "two
        # slots exist" -- a template or tokenizer drift that broke the exact
        # offsets would still pass this and be caught by the hf_cache pin
        # instead (tests/test_loss.py).
        self.inject_positions = [
            i for i, token_id in enumerate(template_ids) if token_id == reserved_id
        ]
        if len(self.inject_positions) != 2:
            raise ValueError(
                f"expected 2 {RESERVED_TOKEN!r} slots in SELFIE_TEMPLATE, found "
                f"{len(self.inject_positions)} (tokenized to {len(template_ids)} tokens)"
            )
        self.template_len = len(template_ids)

        template_ids_tensor = torch.tensor(template_ids, dtype=torch.long)
        with torch.no_grad():
            # (1, template_len, hidden); cloned per batch in __call__.
            self.template_embeds = self.embed_layer(template_ids_tensor.unsqueeze(0))
        self.hidden_size = self.template_embeds.shape[-1]

    @property
    def device(self) -> torch.device:
        return self.template_embeds.device

    def __call__(
        self, vectors: Float[Tensor, "batch hidden"], labels: list[str]
    ) -> tuple[Tensor, dict]:
        batch_size = vectors.shape[0]
        vectors = vectors.to(device=self.device, dtype=torch.float32)
        # Projection stays fp32 ("for training stability", upstream); cast at
        # the embedding-space boundary only.
        soft_tokens = self.projection(vectors)
        model_dtype = self.template_embeds.dtype
        soft_tokens = soft_tokens.to(dtype=model_dtype)

        targets = [target_text(label, self.config) for label in labels]
        target_id_lists = [
            self.tokenizer(text, add_special_tokens=False).input_ids for text in targets
        ]
        target_lens = torch.tensor(
            [len(ids) for ids in target_id_lists], dtype=torch.long
        )
        max_target_len = int(target_lens.max().item())

        template_block = self.template_embeds.expand(batch_size, -1, -1).clone()
        for pos in self.inject_positions:
            template_block[:, pos, :] = soft_tokens

        target_embeds = torch.zeros(
            batch_size,
            max_target_len,
            self.hidden_size,
            dtype=model_dtype,
            device=self.device,
        )
        target_ids = torch.zeros(
            batch_size, max_target_len, dtype=torch.long, device=self.device
        )
        target_mask = torch.zeros(
            batch_size, max_target_len, dtype=torch.long, device=self.device
        )
        for i, ids in enumerate(target_id_lists):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long, device=self.device)
            target_ids[i, :n] = ids_tensor
            target_mask[i, :n] = 1
            with torch.no_grad():
                target_embeds[i, :n] = self.embed_layer(
                    ids_tensor.unsqueeze(0)
                ).squeeze(0)

        full_embeds = torch.cat([template_block, target_embeds], dim=1)
        template_mask = torch.ones(
            batch_size, self.template_len, dtype=torch.long, device=self.device
        )
        full_mask = torch.cat([template_mask, target_mask], dim=1)

        outputs = self.base_model(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            use_cache=False,
        )
        hidden = outputs.last_hidden_state

        # Position template_len - 1 (the template's own last token) predicts
        # the first target token; window k predicts target_ids[:, k].
        window_start = self.template_len - 1
        sliced_hidden = hidden[:, window_start : window_start + max_target_len, :]
        logits = self.lm_head(sliced_hidden)  # (batch, max_target_len, vocab)
        # With a sharded model (device_map="auto") lm_head can land on a
        # different device than the embedding layer self.device follows --
        # everything the loss combines with logits must move to match it.
        target_ids = target_ids.to(logits.device)
        target_mask = target_mask.to(logits.device)

        loss_fn = nn.CrossEntropyLoss(
            reduction="none", label_smoothing=self.config.label_smoothing
        )
        token_losses = loss_fn(
            logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1)
        ).reshape(batch_size, max_target_len)
        clamped = token_losses.clamp(max=self.config.max_loss)
        # Unweighted mean of per-sequence means, not a token-weighted mean --
        # a short target's tokens count for as much as a long one's.
        per_seq_loss = clamped.mul(target_mask).sum(dim=1) / target_mask.sum(
            dim=1
        ).clamp(min=1)
        loss = per_seq_loss.mean()

        with torch.no_grad():
            valid = target_mask.bool()
            soft_token_norms = torch.norm(soft_tokens, p=2, dim=-1)
            stats = {
                "batch_size": batch_size,
                "total_valid_tokens": int(valid.sum().item()),
                "total_clamped_tokens": int(
                    ((token_losses > self.config.max_loss) & valid).sum().item()
                ),
                "max_target_len": max_target_len,
                "mean_target_len": float(target_lens.float().mean().item()),
                "template_len": self.template_len,
                "mean_soft_token_norm": soft_token_norms.mean().item(),
                "max_soft_token_norm": soft_token_norms.max().item(),
            }
        return loss, stats


def subsample(examples: list[Example], n: int, seed: int) -> list[Example]:
    """A fixed, seeded random subsample -- the same mechanism the training
    loop's in-run validation uses.

    :param examples: population to draw from
    :param n: subsample size; the whole population unchanged if `n` exceeds it
    :param seed: RNG seed, so repeated calls with the same inputs agree
    :return: `n` examples, in a fixed order determined by `seed`
    """
    if n >= len(examples):
        return list(examples)
    return random.Random(seed).sample(examples, n)


@torch.no_grad()
def evaluate(
    store, examples: list[Example], scorer: SoftPromptLoss, batch_size: int
) -> dict:
    """Score `examples` in fixed-size batches and average per-batch losses.

    Matches upstream's own `validate()`, which averages per-batch losses over
    batches (equal to averaging per-example except for the last partial
    batch -- a <0.1% discrepancy at 84k examples).

    :return: measured loss, example count, batch count
    """
    total_loss = 0.0
    n_batches = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        vectors = store.vectors[[example.vector_index for example in batch]]
        labels = [example.label for example in batch]
        loss, _ = scorer(vectors, labels)
        total_loss += loss.item()
        n_batches += 1
    measured_loss = total_loss / n_batches if n_batches else float("nan")
    return {
        "measured_loss": measured_loss,
        "n_examples": len(examples),
        "n_batches": n_batches,
    }
