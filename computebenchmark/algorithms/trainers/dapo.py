from dataclasses import dataclass, field

import torch

from .base import BaseTrainer, TrainingConfig
from .grpo import GRPOTrainer


@dataclass
class DAPOConfig(TrainingConfig):
    clip_eps_low: float = 0.2   # clip bound when advantage > 0
    clip_eps_high: float = 0.28  # clip bound when advantage < 0 (looser)
    kl_coeff: float = 0.0        # DAPO removes token-level KL


class DAPOTrainer(GRPOTrainer):
    """DAPO: Decoupled clip + no KL + dynamic sampling (ByteDance, 2025).

    Three changes over GRPO:
    1. Decoupled clipping: separate epsilon for positive vs negative advantages.
    2. KL coefficient set to zero — no reference model penalty.
    3. Dynamic sampling: drop groups where every rollout has the same reward
       (all-correct or all-wrong) since they carry zero gradient signal.
    """

    def __init__(self, config: DAPOConfig):
        if not isinstance(config, DAPOConfig):
            config = DAPOConfig(**vars(config))
        super().__init__(config)
        self.config: DAPOConfig

    def _select_rollouts(
        self,
        prompt_ids: list[torch.Tensor],
        response_ids: list[torch.Tensor],
        rewards: list[float],
    ) -> tuple[list, list, list]:
        G = self.config.group_size
        n_prompts = len(rewards) // G
        kept_p, kept_r, kept_rw = [], [], []
        for i in range(n_prompts):
            group_rewards = rewards[i * G: (i + 1) * G]
            if len(set(group_rewards)) == 1:
                continue  # all same reward — no learning signal
            kept_p.extend(prompt_ids[i * G: (i + 1) * G])
            kept_r.extend(response_ids[i * G: (i + 1) * G])
            kept_rw.extend(group_rewards)
        return kept_p, kept_r, kept_rw

    def _compute_loss(
        self,
        policy_lps: list[torch.Tensor],
        ref_lps: list[torch.Tensor],
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        total = torch.tensor(0.0, device=self.model.device)
        for p_lp, r_lp, adv in zip(policy_lps, ref_lps, advantages):
            log_ratio = p_lp - r_lp.detach()
            ratio = log_ratio.exp()

            # Decoupled clipping: tighter bound on upside, looser on downside
            if adv >= 0:
                clipped = torch.clamp(ratio, 1 - self.config.clip_eps_low, 1 + self.config.clip_eps_low)
            else:
                clipped = torch.clamp(ratio, 1 - self.config.clip_eps_high, 1 + self.config.clip_eps_high)

            pg_loss = -torch.min(ratio * adv, clipped * adv).mean()
            total = total + pg_loss  # no KL term
        return total / len(policy_lps)
