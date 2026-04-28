import json
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
    group_size: int = 8  # rollouts per prompt (G)
    max_prompt_len: int = 512
    max_response_len: int = 512
    num_steps: int = 1000
    eval_every: int = 100
    save_every: int = 500
    lora_rank: int = 16
    lora_alpha: int = 32
    clip_eps: float = 0.2
    kl_coeff: float = 0.01
    output_dir: str = "checkpoints"


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
            self.config.model_id, torch_dtype=torch.bfloat16, device_map="auto"
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
    ) -> dict:
        prompt_ids_list, response_ids_list = self._generate_rollouts(batch_prompts)
        rewards = [
            self._compute_reward(
                self.tokenizer.decode(r_ids, skip_special_tokens=True),
                batch_answers[i // self.config.group_size],
            )
            for i, r_ids in enumerate(response_ids_list)
        ]

        prompt_ids_list, response_ids_list, rewards = self._select_rollouts(
            prompt_ids_list, response_ids_list, rewards
        )
        if not rewards:
            return {"loss": 0.0, "mean_reward": 0.0}

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

        return {"loss": loss.item(), "mean_reward": sum(rewards) / len(rewards)}

    def train(self, dataset, eval_dataset=None) -> list[dict]:
        from ..evaluator import evaluate_gsm8k

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
                step_metrics = self.train_step(batch["prompt"], batch["answer"])
                step_metrics["step"] = step

                if step % self.config.eval_every == 0 and eval_dataset is not None:
                    eval_result = evaluate_gsm8k(
                        self.model, self.tokenizer, max_samples=200
                    )
                    step_metrics.update(eval_result)

                metrics_log.append(step_metrics)
                print(f"step={step} " + " ".join(f"{k}={v:.4f}" for k, v in step_metrics.items() if isinstance(v, float)))

                if step % self.config.save_every == 0:
                    self.model.save_pretrained(out_dir / f"step-{step}")

                step += 1

        with open(log_path, "w") as f:
            for m in metrics_log:
                f.write(json.dumps(m) + "\n")

        self.model.save_pretrained(out_dir / "final")
        return metrics_log
