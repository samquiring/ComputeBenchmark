import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


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
    target_accuracy: float | None = None
    use_vllm: bool = False


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


def gsm8k_accuracy_reward(completions: list[str], answer: list[str], **kwargs) -> list[float]:
    from ...data.gsm8k import extract_answer
    return [1.0 if extract_answer(c) == a else 0.0 for c, a in zip(completions, answer)]


class ConvergenceCallback(TrainerCallback):
    def __init__(
        self,
        config: TrainingConfig,
        eval_dataset,
        tokenizer,
        method_name: str,
    ):
        self.config = config
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.method_name = method_name
        self.tag = f"[{method_name.upper():<10}]"
        self.metrics_log: list[dict] = []
        self.t_start: float | None = None
        self._step_t: float | None = None
        self.convergence: ConvergenceResult | None = None

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        self.t_start = time.perf_counter()
        print(
            f"\n{self.tag} Starting — {self.config.num_steps} steps,"
            f" eval every {self.config.eval_every}",
            flush=True,
        )

    def on_step_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        self._step_t = time.perf_counter()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        step = state.global_step
        step_seconds = time.perf_counter() - self._step_t
        training_elapsed = time.perf_counter() - self.t_start

        step_metrics: dict[str, Any] = {
            "step": step,
            "wall_clock_seconds": training_elapsed,
            "step_seconds": step_seconds,
        }

        # Pull latest training scalars from TRL's log history
        if state.log_history:
            last = state.log_history[-1]
            for k, v in last.items():
                if k not in ("step", "epoch", "total_flos"):
                    step_metrics[k] = v

        tokens_seen = getattr(state, "num_input_tokens_seen", 0) or 0
        step_metrics["tokens_seen"] = tokens_seen

        run_eval = (
            step % self.config.eval_every == 0
            and step > 0
            and self.eval_dataset is not None
        )
        if run_eval:
            from ..evaluator import evaluate_gsm8k

            t_eval = time.perf_counter()
            model.eval()
            eval_result = evaluate_gsm8k(
                model, self.tokenizer, max_samples=100, max_new_tokens=256
            )
            model.train()
            eval_seconds = time.perf_counter() - t_eval
            # Shift start forward so eval time is excluded from wall clock
            self.t_start += eval_seconds
            step_metrics.update(eval_result)
            step_metrics["eval_seconds"] = eval_seconds

            acc = eval_result.get("gsm8k_accuracy", 0.0)
            elapsed = step_metrics["wall_clock_seconds"]
            target_str = (
                f"  target={self.config.target_accuracy:.3f}"
                if self.config.target_accuracy
                else ""
            )
            print(
                f"{self.tag} step={step:>4}/{self.config.num_steps}"
                f"  acc={acc:.3f}{target_str}"
                f"  tokens={tokens_seen:,}  elapsed={elapsed / 60:.1f}min",
                flush=True,
            )

            if (
                self.config.target_accuracy is not None
                and self.convergence is None
                and acc >= self.config.target_accuracy
            ):
                self.convergence = ConvergenceResult(
                    method=self.method_name,
                    target_accuracy=self.config.target_accuracy,
                    reached_target=True,
                    steps_to_target=step,
                    wall_clock_seconds_to_target=elapsed,
                    tokens_seen_to_target=tokens_seen,
                    final_accuracy=acc,
                    final_step=step,
                    final_wall_clock_seconds=elapsed,
                )
                print(
                    f"{self.tag} *** TARGET {self.config.target_accuracy:.3f} REACHED"
                    f" — step={step} elapsed={elapsed / 60:.1f}min ***",
                    flush=True,
                )
                control.should_training_stop = True
        else:
            loss = step_metrics.get("loss", 0.0)
            reward = step_metrics.get("reward", step_metrics.get("mean_reward", 0.0))
            print(
                f"{self.tag} step={step:>4}/{self.config.num_steps}"
                f"  loss={loss:.4f}  reward={reward:.3f}"
                f"  elapsed={training_elapsed / 60:.1f}min",
                flush=True,
            )

        self.metrics_log.append(step_metrics)

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        if self.convergence is not None:
            return
        elapsed = time.perf_counter() - self.t_start
        final_acc = next(
            (m["gsm8k_accuracy"] for m in reversed(self.metrics_log) if "gsm8k_accuracy" in m),
            0.0,
        )
        self.convergence = ConvergenceResult(
            method=self.method_name,
            target_accuracy=self.config.target_accuracy or 0.0,
            reached_target=False,
            steps_to_target=None,
            wall_clock_seconds_to_target=None,
            tokens_seen_to_target=None,
            final_accuracy=final_acc,
            final_step=state.global_step,
            final_wall_clock_seconds=elapsed,
        )


def make_lora_config(config: TrainingConfig) -> LoraConfig:
    return LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )


class BaseTRLWrapper(ABC):
    def __init__(self, config: TrainingConfig):
        self.config = config

    @abstractmethod
    def _make_trainer(self, train_dataset, eval_dataset, callback: ConvergenceCallback):
        ...

    def train(self, dataset, eval_dataset=None) -> tuple[list[dict], ConvergenceResult | None]:
        from transformers import AutoTokenizer

        method_name = (
            self.__class__.__name__.replace("Trainer", "").replace("Wrapper", "").lower()
        )
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id, padding_side="left"
        )
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        callback = ConvergenceCallback(self.config, eval_dataset, tokenizer, method_name)
        trainer = self._make_trainer(dataset, eval_dataset, callback)
        trainer.train()

        log_path = out_dir / "metrics.jsonl"
        with open(log_path, "w") as f:
            for m in callback.metrics_log:
                f.write(json.dumps(m) + "\n")

        return callback.metrics_log, callback.convergence
