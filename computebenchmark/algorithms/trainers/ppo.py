import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig
try:
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
except ImportError:
    try:
        from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead
        from trl import PPOConfig, PPOTrainer
    except ImportError:
        import trl as _trl
        raise ImportError(
            f"AutoModelForCausalLMWithValueHead not found in TRL {_trl.__version__}. "
            "Downgrade with: pip install 'trl>=0.9.0,<0.15.0'"
        )
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
            ppo_epochs=getattr(self.config, "ppo_epochs", 4),
            init_kl_coef=getattr(self.config, "init_kl_coef", 0.2),
            target=getattr(self.config, "target_kl", 6.0),
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

        from .base import ConvergenceResult

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "metrics.jsonl"
        metrics_log = []
        tokens_seen = 0
        convergence = None
        t_start = time.perf_counter()
        step = 0

        while step < self.config.num_steps:
            for i in range(0, len(dataset), self.config.batch_size):
                if step >= self.config.num_steps:
                    break

                t_step = time.perf_counter()
                batch = dataset[i: i + self.config.batch_size]
                queries = [
                    self.tokenizer.encode(p, return_tensors="pt").squeeze()
                    for p in batch["prompt"]
                ]
                responses = trainer.generate(
                    queries, max_new_tokens=self.config.max_response_len
                )
                reward_vals = [
                    1.0 if extract_answer(
                        self.tokenizer.decode(r, skip_special_tokens=True)
                    ) == ans else 0.0
                    for r, ans in zip(responses, batch["answer"])
                ]
                tokens_seen += sum(len(q) + len(r) for q, r in zip(queries, responses))

                stats = trainer.step(queries, responses, [torch.tensor(v) for v in reward_vals])
                elapsed = time.perf_counter() - t_start
                step_metrics = {
                    "step": step,
                    "loss": stats.get("ppo/loss/total", 0.0),
                    "mean_reward": sum(reward_vals) / len(reward_vals),
                    "reward_std": torch.tensor(reward_vals).std().item(),
                    "tokens_seen": tokens_seen,
                    "wall_clock_seconds": elapsed,
                    "step_seconds": time.perf_counter() - t_step,
                }

                if step % self.config.eval_every == 0 and eval_dataset is not None:
                    eval_result = evaluate_gsm8k(model, self.tokenizer, max_samples=200)
                    step_metrics.update(eval_result)
                    acc = step_metrics.get("gsm8k_accuracy", 0.0)
                    print(
                        f"  step={step:>4}  acc={acc:.3f}  reward={step_metrics['mean_reward']:.3f}"
                        f"  tokens={tokens_seen:,}  elapsed={elapsed/60:.1f}min",
                        flush=True,
                    )
                    if (
                        self.config.target_accuracy is not None
                        and convergence is None
                        and acc >= self.config.target_accuracy
                    ):
                        convergence = ConvergenceResult(
                            method="ppo",
                            target_accuracy=self.config.target_accuracy,
                            reached_target=True,
                            steps_to_target=step,
                            wall_clock_seconds_to_target=elapsed,
                            tokens_seen_to_target=tokens_seen,
                            final_accuracy=acc,
                            final_step=step,
                            final_wall_clock_seconds=elapsed,
                        )
                        print(f"  *** Target {self.config.target_accuracy:.3f} reached at step {step} ({elapsed/60:.1f}min) ***", flush=True)
                else:
                    print(f"  step={step:>4}  reward={step_metrics['mean_reward']:.3f}  elapsed={elapsed/60:.1f}min", flush=True)

                metrics_log.append(step_metrics)

                if step % self.config.save_every == 0 and step > 0:
                    model.save_pretrained(out_dir / f"step-{step}")

                step += 1

        elapsed_total = time.perf_counter() - t_start
        final_acc = next(
            (m["gsm8k_accuracy"] for m in reversed(metrics_log) if "gsm8k_accuracy" in m), 0.0
        )
        if convergence is None:
            convergence = ConvergenceResult(
                method="ppo",
                target_accuracy=self.config.target_accuracy or 0.0,
                reached_target=False,
                steps_to_target=None,
                wall_clock_seconds_to_target=None,
                tokens_seen_to_target=None,
                final_accuracy=final_acc,
                final_step=step,
                final_wall_clock_seconds=elapsed_total,
            )

        with open(log_path, "w") as f:
            for m in metrics_log:
                f.write(json.dumps(m) + "\n")

        model.save_pretrained(out_dir / "final")
        return metrics_log, convergence
