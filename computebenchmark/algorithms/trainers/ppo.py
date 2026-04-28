"""REINFORCE baseline trainer.

Replaces TRL's PPO as the baseline. REINFORCE is the simplest policy gradient
algorithm: no critic, no group normalization, no clipping. A moving average of
recent rewards serves as the variance-reducing baseline.

This gives a cleaner paper story: REINFORCE → GRPO → DAPO is a monotonic
progression of algorithmic improvements, each adding one mechanism.
"""

import torch

from .base import BaseTrainer, TrainingConfig


class PPOTrainerWrapper(BaseTrainer):
    """REINFORCE with a moving-average reward baseline.

    Named PPOTrainerWrapper to keep the registry key 'ppo' unchanged so
    existing commands don't break, but the implementation is TRL-free.
    """

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self._reward_baseline = 0.0
        self._baseline_momentum = 0.99

    def _compute_advantages(self, rewards: list[float]) -> torch.Tensor:
        # Update moving average baseline
        for r in rewards:
            self._reward_baseline = (
                self._baseline_momentum * self._reward_baseline
                + (1 - self._baseline_momentum) * r
            )
        return torch.tensor(
            [r - self._reward_baseline for r in rewards], dtype=torch.float32
        )

    def _compute_loss(
        self,
        policy_lps: list[torch.Tensor],
        ref_lps: list[torch.Tensor],
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        total = torch.tensor(0.0, device=self.model.device)
        for p_lp, adv in zip(policy_lps, advantages):
            pg_loss = -(p_lp.mean() * adv)
            total = total + pg_loss
        return total / len(policy_lps)
