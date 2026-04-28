import math
import torch
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from .base import BaseTRLWrapper, ConvergenceCallback, gsm8k_accuracy_reward, make_lora_config


def _setattr_first(obj, value, *names):
    """Set the first attribute name that exists on obj."""
    for name in names:
        if hasattr(obj, name):
            setattr(obj, name, value)
            return


class GRPOTrainerWrapper(BaseTRLWrapper):
    """Group Relative Policy Optimization (DeepSeekMath, 2024).

    Advantages normalized within each group of G rollouts; no critic needed.
    """

    def _load_model(self) -> AutoModelForCausalLM:
        """Load base model with best available flash attention; PEFT applied by TRL."""
        load_kwargs = dict(torch_dtype=torch.bfloat16)
        for impl in ("flash_attention_3", "flash_attention_2"):
            try:
                return AutoModelForCausalLM.from_pretrained(
                    self.config.model_id, attn_implementation=impl, **load_kwargs
                )
            except (ImportError, ValueError):
                continue
        return AutoModelForCausalLM.from_pretrained(self.config.model_id, **load_kwargs)

    def _make_grpo_config(self) -> GRPOConfig:
        c = self.config
        gen_batch = math.ceil(c.batch_size / c.group_size) * c.group_size
        # Construct with only output_dir to avoid version-specific __init__ rejections,
        # then set everything via attribute assignment.
        cfg = GRPOConfig(output_dir=c.output_dir)
        cfg.per_device_train_batch_size = c.batch_size
        cfg.num_generations = c.group_size
        cfg.max_steps = c.num_steps
        cfg.learning_rate = c.learning_rate
        cfg.beta = c.kl_coeff
        cfg.epsilon = c.clip_eps
        cfg.bf16 = True
        cfg.gradient_checkpointing = True
        cfg.use_vllm = c.use_vllm
        cfg.save_steps = c.save_every
        cfg.logging_steps = 1
        cfg.report_to = "none"
        cfg.dataloader_pin_memory = False
        # Field names vary by TRL version
        if hasattr(cfg, "generation_batch_size"):
            cfg.generation_batch_size = gen_batch
        _setattr_first(cfg, c.max_prompt_len, "max_prompt_length", "query_length")
        _setattr_first(cfg, c.max_response_len, "max_completion_length", "response_length", "max_new_tokens")
        return cfg

    def _reward_funcs(self) -> list:
        return [gsm8k_accuracy_reward]

    def _make_trainer(self, train_dataset, eval_dataset, callback: ConvergenceCallback):
        return GRPOTrainer(
            model=self._load_model(),
            args=self._make_grpo_config(),
            train_dataset=train_dataset,
            reward_funcs=self._reward_funcs(),
            peft_config=make_lora_config(self.config),
            callbacks=[callback],
        )
