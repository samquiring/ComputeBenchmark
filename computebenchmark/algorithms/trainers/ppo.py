"""REINFORCE baseline trainer (registered as 'ppo' for CLI compatibility).

Uses GRPOTrainer with tight epsilon (no effective clipping) and no KL penalty
to approximate vanilla REINFORCE with group-relative advantage normalization.
This gives a cleaner paper baseline than custom REINFORCE: GRPO baseline → GRPO → DAPO.
"""

from trl import GRPOConfig

from .grpo import GRPOTrainerWrapper


class PPOTrainerWrapper(GRPOTrainerWrapper):
    """REINFORCE-style baseline using GRPOTrainer.

    Clipping disabled (epsilon=1.0 so ratio bounds never bite),
    KL penalty removed (beta=0). Pure policy gradient with group baseline.
    """

    def _make_grpo_config(self) -> GRPOConfig:
        c = self.config
        cfg = super()._make_grpo_config()
        cfg.beta = 0.0    # no KL penalty
        cfg.epsilon = 1.0  # clip bounds [0, 2] — never active
        return cfg
