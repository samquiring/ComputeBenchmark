"""DAPO: Decoupled Clip + Dynamic Sampling Policy Optimization (ByteDance, 2025).

Three changes over GRPO:
1. Decoupled clipping: epsilon_low for positive advantages, epsilon_high for negative.
2. KL coefficient zero — no reference model penalty.
3. Dynamic sampling: groups where all rollouts share the same reward are dropped.
"""

from trl import GRPOConfig

from .grpo import GRPOTrainerWrapper


class DAPOTrainer(GRPOTrainerWrapper):
    def _make_grpo_config(self) -> GRPOConfig:
        cfg = super()._make_grpo_config()
        cfg.beta = 0.0     # DAPO removes KL penalty
        cfg.epsilon = 0.2  # lower clip bound (positive advantages)
        if hasattr(cfg, "epsilon_high"):
            cfg.epsilon_high = 0.28   # decoupled clipping (trl >= 0.15)
        if hasattr(cfg, "use_dr_grpo"):
            cfg.use_dr_grpo = True    # dynamic sampling (trl >= 0.15)
        return cfg
