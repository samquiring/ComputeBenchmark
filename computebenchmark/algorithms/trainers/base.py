import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TrainingConfig:
    model_id: str
    learning_rate: float = 5e-6
    batch_size: int = 4
    group_size: int = 8
    max_prompt_len: int = 512
    max_response_len: int = 512
    num_steps: int = 500
    eval_every: int = 50
    save_every: int = 500
    lora_rank: int = 16
    lora_alpha: int = 32
    clip_eps: float = 0.2
    kl_coeff: float = 0.01
    output_dir: str = "checkpoints"
    target_accuracy: float | None = None  # stop early when this GSM8K accuracy is reached


@dataclass
class ConvergenceResult:
    method: str
    target_accuracy: float
    reached_target: bool
    steps_to_target: int | None
    wall_clock_seconds_to_target: float | None
    tokens_seen_to_target: int | None
    final_accuracy: float
    final_step: int
    final_wall_clock_seconds: float


class BaseTrainer(ABC):
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, padding_side="left"
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self._load_model()
        self.ref_model = self._load_ref_model()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate
        )

    def _load_model(self) -> AutoModelForCausalLM:
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_3",
        )
        lora_cfg = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        return get_peft_model(model, lora_cfg)

    def _load_ref_model(self) -> AutoModelForCausalLM:
        ref = AutoModelForCausalLM.from_pretrained(
            self.config.model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        return ref

    @torch.no_grad()
    def _generate_rollouts(
        self, prompts: list[str]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        prompt_ids_list, response_ids_list = [], []
        for prompt in prompts:
            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.config.max_prompt_len,
                truncation=True,
            ).to(self.model.device)
            for _ in range(self.config.group_size):
                out = self.model.generate(
                    **enc,
                    max_new_tokens=self.config.max_response_len,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                response_ids = out[0][enc["input_ids"].shape[1]:]
                prompt_ids_list.append(enc["input_ids"][0])
                response_ids_list.append(response_ids)
        return prompt_ids_list, response_ids_list

    def _compute_log_probs(
        self,
        model: AutoModelForCausalLM,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0)
        with torch.set_grad_enabled(model.training):
            logits = model(input_ids).logits
        prompt_len = len(prompt_ids)
        response_logits = logits[0, prompt_len - 1: prompt_len + len(response_ids) - 1]
        log_probs = F.log_softmax(response_logits, dim=-1)
        return log_probs.gather(1, response_ids.unsqueeze(1)).squeeze(1)

    def _compute_reward(self, response: str, ground_truth: str) -> float:
        from ...data.gsm8k import extract_answer
        predicted = extract_answer(response)
        return 1.0 if predicted is not None and predicted == ground_truth else 0.0

    def _tokens_in_batch(self, prompt_ids: list[torch.Tensor], response_ids: list[torch.Tensor]) -> int:
        return sum(len(p) + len(r) for p, r in zip(prompt_ids, response_ids))

    @abstractmethod
    def _compute_advantages(self, rewards: list[float]) -> torch.Tensor:
        ...

    @abstractmethod
    def _compute_loss(
        self,
        policy_lps: list[torch.Tensor],
        ref_lps: list[torch.Tensor],
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def _select_rollouts(
        self,
        prompt_ids: list[torch.Tensor],
        response_ids: list[torch.Tensor],
        rewards: list[float],
    ) -> tuple[list, list, list]:
        return prompt_ids, response_ids, rewards

    def train_step(
        self, batch_prompts: list[str], batch_answers: list[str]
    ) -> tuple[dict, int]:
        prompt_ids_list, response_ids_list = self._generate_rollouts(batch_prompts)
        rewards = [
            self._compute_reward(
                self.tokenizer.decode(r_ids, skip_special_tokens=True),
                batch_answers[i // self.config.group_size],
            )
            for i, r_ids in enumerate(response_ids_list)
        ]

        tokens = self._tokens_in_batch(prompt_ids_list, response_ids_list)

        prompt_ids_list, response_ids_list, rewards = self._select_rollouts(
            prompt_ids_list, response_ids_list, rewards
        )
        if not rewards:
            return {"loss": 0.0, "mean_reward": 0.0, "reward_std": 0.0}, tokens

        policy_lps, ref_lps = [], []
        for p_ids, r_ids in zip(prompt_ids_list, response_ids_list):
            policy_lps.append(self._compute_log_probs(self.model, p_ids, r_ids))
            ref_lps.append(self._compute_log_probs(self.ref_model, p_ids, r_ids))

        advantages = self._compute_advantages(rewards)
        loss = self._compute_loss(policy_lps, ref_lps, advantages)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        reward_t = torch.tensor(rewards, dtype=torch.float32)
        return {
            "loss": loss.item(),
            "mean_reward": reward_t.mean().item(),
            "reward_std": reward_t.std().item(),
        }, tokens

    def train(self, dataset, eval_dataset=None) -> tuple[list[dict], ConvergenceResult | None]:
        from ..evaluator import evaluate_gsm8k

        method_name = self.__class__.__name__.replace("Trainer", "").replace("Wrapper", "").lower()
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "metrics.jsonl"

        tag = f"[{method_name.upper():<10}]"
        metrics_log = []
        tokens_seen = 0
        convergence: ConvergenceResult | None = None
        t_start = time.perf_counter()
        step = 0

        print(f"\n{tag} Starting — {self.config.num_steps} steps, eval every {self.config.eval_every}", flush=True)

        while step < self.config.num_steps:
            for i in range(0, len(dataset), self.config.batch_size):
                if step >= self.config.num_steps:
                    break

                t_step = time.perf_counter()
                batch = dataset[i: i + self.config.batch_size]
                step_metrics, step_tokens = self.train_step(batch["prompt"], batch["answer"])
                tokens_seen += step_tokens

                step_metrics["step"] = step
                step_metrics["tokens_seen"] = tokens_seen
                training_elapsed = time.perf_counter() - t_start
                step_metrics["wall_clock_seconds"] = training_elapsed
                step_metrics["step_seconds"] = time.perf_counter() - t_step

                if step % self.config.eval_every == 0 and eval_dataset is not None:
                    t_eval = time.perf_counter()
                    eval_result = evaluate_gsm8k(self.model, self.tokenizer, max_samples=100, max_new_tokens=256)
                    eval_seconds = time.perf_counter() - t_eval
                    t_start += eval_seconds  # shift start forward so eval time is excluded
                    step_metrics.update(eval_result)
                    step_metrics["eval_seconds"] = eval_seconds

                    acc = step_metrics.get("gsm8k_accuracy", 0.0)
                    elapsed = training_elapsed
                    target_str = f"  target={self.config.target_accuracy:.3f}" if self.config.target_accuracy else ""
                    print(
                        f"{tag} step={step:>4}/{self.config.num_steps}"
                        f"  acc={acc:.3f}{target_str}"
                        f"  reward={step_metrics['mean_reward']:.3f}"
                        f"  tokens={tokens_seen:,}  elapsed={elapsed/60:.1f}min",
                        flush=True,
                    )

                    if (
                        self.config.target_accuracy is not None
                        and convergence is None
                        and acc >= self.config.target_accuracy
                    ):
                        convergence = ConvergenceResult(
                            method=method_name,
                            target_accuracy=self.config.target_accuracy,
                            reached_target=True,
                            steps_to_target=step,
                            wall_clock_seconds_to_target=elapsed,
                            tokens_seen_to_target=tokens_seen,
                            final_accuracy=acc,
                            final_step=step,
                            final_wall_clock_seconds=elapsed,
                        )
                        print(f"{tag} *** TARGET {self.config.target_accuracy:.3f} REACHED — step={step} elapsed={elapsed/60:.1f}min ***", flush=True)
                else:
                    print(
                        f"{tag} step={step:>4}/{self.config.num_steps}"
                        f"  reward={step_metrics['mean_reward']:.3f}"
                        f"  std={step_metrics['reward_std']:.3f}"
                        f"  elapsed={step_metrics['wall_clock_seconds']/60:.1f}min",
                        flush=True,
                    )

                metrics_log.append(step_metrics)

                if step % self.config.save_every == 0 and step > 0:
                    self.model.save_pretrained(out_dir / f"step-{step}")

                step += 1

        elapsed_total = time.perf_counter() - t_start
        final_acc = next(
            (m["gsm8k_accuracy"] for m in reversed(metrics_log) if "gsm8k_accuracy" in m), 0.0
        )

        if convergence is None:
            convergence = ConvergenceResult(
                method=method_name,
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

        self.model.save_pretrained(out_dir / "final")
        return metrics_log, convergence
