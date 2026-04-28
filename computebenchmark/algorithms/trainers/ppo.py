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
        return GRPOConfig(
            output_dir=c.output_dir,
            per_device_train_batch_size=c.batch_size,
            num_generations=c.group_size,
            max_prompt_length=c.max_prompt_len,
            max_completion_length=c.max_response_len,
            max_steps=c.num_steps,
            learning_rate=c.learning_rate,
            beta=0.0,      # no KL penalty
            epsilon=1.0,   # clip bounds [0, 2] — never active
            bf16=True,
            gradient_checkpointing=True,
            use_vllm=c.use_vllm,
            save_steps=c.save_every,
            logging_steps=1,
            report_to="none",
            dataloader_pin_memory=False,
        )
