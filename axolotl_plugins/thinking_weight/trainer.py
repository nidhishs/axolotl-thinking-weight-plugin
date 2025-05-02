import logging
from typing import Any, Optional

import torch
import torch.nn.functional as F
from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.prompters import IGNORE_TOKEN_ID
from transformers import PreTrainedTokenizerBase

LOG = logging.getLogger("axolotl_plugins.thinking_weight.trainer")
EPS = 1e-9


class ThinkingWeightTrainer(AxolotlTrainer):
    """
    Trainer that applies different weights to thinking vs answer tokens during loss calculation.
    We identify thinking tokens based on guaranteed single-token thinking tags.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialized by callback `add_callbacks_post_trainer`.
        self.use_thinking_weight: bool = None
        self.thinking_token_weight: float = None
        self.answer_token_weight: float = None
        self.think_start_token: str = None
        self.think_end_token: str = None

        self.think_start_token_id: Optional[int] = None
        self.think_end_token_id: Optional[int] = None

    def init_thinking_token_ids(self):
        """
        Initialize the thinking start and end token IDs based on the tokenizer.
        """
        if self.think_start_token == self.think_end_token:
            raise ValueError(
                "Thinking start and end tokens must be different. "
                f"Got {self.think_start_token} and {self.think_end_token}."
            )

        if not isinstance(self.processing_class, PreTrainedTokenizerBase):
            raise ValueError(
                "Thinking weight trainer requires a tokenizer that is a subclass of PreTrainedTokenizerBase. "
                f"Got {type(self.processing_class)}."
            )

        self.think_start_token_id = self.processing_class.encode(
            self.think_start_token, add_special_tokens=False
        )
        self.think_end_token_id = self.processing_class.encode(
            self.think_end_token, add_special_tokens=False
        )
        try:
            self.think_start_token_id = _require_single_token(
                self.think_start_token, self.think_start_token_id
            )
            self.think_end_token_id = _require_single_token(
                self.think_end_token, self.think_end_token_id
            )
            LOG.info(
                "Thinking start token ID: %s, end token ID: %s. Thinking weight: %s, Answer weight: %s",
                self.think_start_token_id,
                self.think_end_token_id,
                self.thinking_token_weight,
                self.answer_token_weight,
            )
        except Exception as e:
            LOG.error("Disabling thinking-weight logic: %s", e)
            self.use_thinking_weight = False

    def compute_loss(
        self, model, inputs: dict, return_outputs: bool = False, num_items_in_batch=None
    ):
        if not self.use_thinking_weight:
            return super().compute_loss(
                model, inputs, return_outputs, num_items_in_batch
            )

        labels = inputs.pop("labels")
        input_ids = inputs["input_ids"]

        # Forward pass
        outputs = model(**inputs)
        logits = outputs["logits"]  # (batch_size, seq_len, vocab_size)
        vocab_size = logits.size(-1)

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_inputs = input_ids[..., :-1].contiguous()

        # Flatten
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        flat_inputs = shift_inputs.view(-1)

        # Create weights
        weights = torch.full_like(
            flat_labels, self.answer_token_weight, dtype=flat_logits.dtype
        )

        if (flat_inputs == self.think_start_token_id).any():
            tag_start = flat_inputs == self.think_start_token_id
            tag_end = flat_inputs == self.think_end_token_id
            shift_tag_end = torch.zeros_like(tag_end)
            shift_tag_end[1:] = tag_end[:-1]
            inside = (
                torch.cumsum(
                    tag_start.to(torch.int8) - shift_tag_end.to(torch.int8), dim=0
                )
                > 0
            )
            weights[inside] = self.thinking_token_weight

        ignored = flat_labels == IGNORE_TOKEN_ID
        weights[ignored] = 0.0

        # Compute loss
        loss = _weighted_cross_entropy(flat_logits, flat_labels, weights)

        return (loss, outputs) if return_outputs else loss


def _require_single_token(tag: str, ids: list[int]) -> int:
    """
    Check if the tag is a single token and return its ID.

    Args:
        tag (str): The tag to check.
        ids (list[int]): The list of token IDs.
    Raises:
        ValueError: If the tag is not a single token.
    Returns:
        int: The ID of the single token.
    """
    if len(ids) != 1:
        raise ValueError(
            f"Tag '{tag}' is not a single token. Found {len(ids)} tokens: {ids}\n"
            "Ensure it's a single token (add via `tokens`)."
        )
    return ids[0]


def _weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
):
    """
    Compute the weighted cross-entropy loss.
    """
    loss_unreduced = F.cross_entropy(
        logits, labels, reduction="none", ignore_index=IGNORE_TOKEN_ID
    )
    loss = (loss_unreduced * weights).sum() / (weights.sum() + EPS)
    return loss
