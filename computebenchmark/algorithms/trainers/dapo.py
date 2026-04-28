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
        c = self.config
        cfg = GRPOConfig(
            output_dir=c.output_dir,
            per_device_train_batch_size=c.batch_size,
            num_generations=c.group_size,
            max_prompt_length=c.max_prompt_len,
            max_completion_length=c.max_response_len,
            max_steps=c.num_steps,
            learning_rate=c.learning_rate,
            beta=0.0,          # DAPO removes KL penalty
            epsilon=0.2,       # lower clip bound (positive advantages)
            bf16=True,
            gradient_checkpointing=True,
            use_vllm=c.use_vllm,
            save_steps=c.save_every,
            logging_steps=1,
            report_to="none",
            dataloader_pin_memory=False,
        )
        # epsilon_high for decoupled clipping (trl >= 0.15); silently ignored on older versions
        if hasattr(cfg, "epsilon_high"):
            cfg.epsilon_high = 0.28
        # Drop all-correct / all-wrong groups (dynamic sampling); trl >= 0.15
        if hasattr(cfg, "use_dr_grpo"):
            cfg.use_dr_grpo = True
        return cfg
