import torch

from .base import BaseTrainer, TrainingConfig


class GRPOTrainer(BaseTrainer):
    """Group Relative Policy Optimization (DeepSeekMath, 2024).

    Advantages are normalized within each group of G rollouts for the same prompt,
    removing the need for a separate critic network.
    """

    def _compute_advantages(self, rewards: list[float]) -> torch.Tensor:
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        G = self.config.group_size
        grouped = rewards_t.reshape(-1, G)
        mean = grouped.mean(dim=1, keepdim=True)
        std = grouped.std(dim=1, keepdim=True)
        return ((grouped - mean) / (std + 1e-8)).reshape(-1)

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
            clipped = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps)
            pg_loss = -torch.min(ratio * adv, clipped * adv).mean()
            kl = (r_lp.detach().exp() * (r_lp.detach() - p_lp)).sum()
            total = total + pg_loss + self.config.kl_coeff * kl
        return total / len(policy_lps)
