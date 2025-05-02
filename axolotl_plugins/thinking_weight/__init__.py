import logging

from axolotl.integrations.base import BasePlugin

from .args import ThinkingWeightArgs  # pylint: disable=unused-import. # noqa: F401
from .trainer import ThinkingWeightTrainer

LOG = logging.getLogger("axolotl_plugins.thinking_weight")


class ThinkingWeightPlugin(BasePlugin):
    """
    Enable weighted loss for thinking tokens.
    """

    def get_input_args(self):
        return "axolotl_plugins.thinking_weight.ThinkingWeightArgs"

    def get_trainer_cls(self, cfg):
        if cfg.get("use_thinking_weight", False):

            LOG.info("Using ThinkingWeightTrainer for training.")
            return ThinkingWeightTrainer
        return None

    def add_callbacks_post_trainer(self, cfg: dict, trainer: ThinkingWeightTrainer):
        trainer.use_thinking_weight = cfg.get("use_thinking_weight", False)
        trainer.thinking_token_weight = cfg.get("thinking_token_weight", 1.0)
        trainer.answer_token_weight = cfg.get("answer_token_weight", 1.5)
        trainer.think_start_token = cfg.get("think_start_token", "<think>")
        trainer.think_end_token = cfg.get("think_end_token", "</think>")
        trainer.init_thinking_token_ids()

        return []
