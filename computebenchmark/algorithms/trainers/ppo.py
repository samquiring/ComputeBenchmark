import json
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from transformers import AutoTokenizer

from .base import TrainingConfig
from ...data.gsm8k import build_dataset, extract_answer


@dataclass
class PPOTrainerConfig(TrainingConfig):
    ppo_epochs: int = 4
    init_kl_coef: float = 0.2
    target_kl: float = 6.0


class PPOTrainerWrapper:
    """PPO baseline using TRL's PPOTrainer with a value head.

    PPO requires a separate critic (value head) to estimate V(s) for GAE
    advantages, which is the main architectural difference from GRPO/DAPO.
    We use TRL's implementation to avoid reimplementing GAE from scratch.
    """

    def __init__(self, config: PPOTrainerConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, padding_side="left"
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(self, dataset, eval_dataset=None) -> list[dict]:
        from ..evaluator import evaluate_gsm8k

        ppo_config = PPOConfig(
            model_name=self.config.model_id,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mini_batch_size=max(1, self.config.batch_size // 2),
            ppo_epochs=self.config.ppo_epochs,
            init_kl_coef=self.config.init_kl_coef,
            target=self.config.target_kl,
            log_with=None,
        )

        lora_cfg = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )

        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            peft_config=lora_cfg,
        )

        trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=self.tokenizer)

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "metrics.jsonl"
        metrics_log = []
        step = 0

        while step < self.config.num_steps:
            for i in range(0, len(dataset), self.config.batch_size):
                if step >= self.config.num_steps:
                    break

                batch = dataset[i: i + self.config.batch_size]
                queries = [
                    self.tokenizer.encode(p, return_tensors="pt").squeeze()
                    for p in batch["prompt"]
                ]
                responses = trainer.generate(
                    queries, max_new_tokens=self.config.max_response_len
                )
                rewards = [
                    torch.tensor(
                        1.0 if extract_answer(
                            self.tokenizer.decode(r, skip_special_tokens=True)
                        ) == ans else 0.0
                    )
                    for r, ans in zip(responses, batch["answer"])
                ]

                stats = trainer.step(queries, responses, rewards)
                step_metrics = {
                    "step": step,
                    "loss": stats.get("ppo/loss/total", 0.0),
                    "mean_reward": sum(r.item() for r in rewards) / len(rewards),
                }

                if step % self.config.eval_every == 0 and eval_dataset is not None:
                    eval_result = evaluate_gsm8k(model, self.tokenizer, max_samples=200)
                    step_metrics.update(eval_result)

                metrics_log.append(step_metrics)
                print(f"step={step} " + " ".join(f"{k}={v:.4f}" for k, v in step_metrics.items() if isinstance(v, float)))

                if step % self.config.save_every == 0:
                    model.save_pretrained(out_dir / f"step-{step}")

                step += 1

        with open(log_path, "w") as f:
            for m in metrics_log:
                f.write(json.dumps(m) + "\n")

        model.save_pretrained(out_dir / "final")
        return metrics_log
