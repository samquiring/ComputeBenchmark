from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="ComputeBenchmark: hardware throughput and RL algorithm comparison.")
compute_app = typer.Typer(help="Section 1: hardware throughput benchmarks.")
algo_app = typer.Typer(help="Section 2: RL algorithm comparison.")
app.add_typer(compute_app, name="compute")
app.add_typer(algo_app, name="algorithms")


@compute_app.command("run")
def compute_run(
    model_id: Annotated[str, typer.Option("--model-id", "-m", help="HuggingFace model ID")],
    output: Annotated[str, typer.Option("--output", "-o")] = "results/compute.json",
    batch_sizes: Annotated[str, typer.Option(help="Comma-separated, e.g. 1,4,8")] = "1,4,8",
    prompt_lengths: Annotated[str, typer.Option(help="Comma-separated, e.g. 128,512,1024")] = "128,512,1024",
    generation_length: Annotated[int, typer.Option()] = 256,
    warmup_iters: Annotated[int, typer.Option()] = 3,
    bench_iters: Annotated[int, typer.Option()] = 10,
    dtype: Annotated[str, typer.Option()] = "bfloat16",
    csv: Annotated[bool, typer.Option("--csv")] = False,
):
    from .compute.runner import ComputeConfig, run
    from .compute.report import print_table, save_csv, save_json

    config = ComputeConfig(
        model_id=model_id,
        batch_sizes=[int(x) for x in batch_sizes.split(",")],
        prompt_lengths=[int(x) for x in prompt_lengths.split(",")],
        generation_length=generation_length,
        warmup_iters=warmup_iters,
        bench_iters=bench_iters,
        dtype=dtype,
    )

    typer.echo(f"Running compute benchmark for {model_id}...")
    results = run(config)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, out_path)
    if csv:
        save_csv(results, out_path.with_suffix(".csv"))

    print_table(results)
    typer.echo(f"\nSaved to {out_path}")


@algo_app.command("train")
def algo_train(
    method: Annotated[str, typer.Argument(help="ppo | grpo | dapo | rlvr")],
    model_id: Annotated[str, typer.Option("--model-id", "-m")],
    steps: Annotated[int, typer.Option("--steps")] = 1000,
    batch_size: Annotated[int, typer.Option()] = 4,
    group_size: Annotated[int, typer.Option(help="Rollouts per prompt (G)")] = 8,
    eval_every: Annotated[int, typer.Option()] = 100,
    lora_rank: Annotated[int, typer.Option()] = 16,
    output_dir: Annotated[str, typer.Option("--output-dir", "-o")] = "checkpoints",
):
    from .algorithms.trainers import TRAINERS
    from .algorithms.trainers.base import TrainingConfig
    from .data.gsm8k import build_dataset
    from transformers import AutoTokenizer

    if method not in TRAINERS:
        typer.echo(f"Unknown method '{method}'. Choose from: {', '.join(TRAINERS)}")
        raise typer.Exit(1)

    config = TrainingConfig(
        model_id=model_id,
        num_steps=steps,
        batch_size=batch_size,
        group_size=group_size,
        eval_every=eval_every,
        lora_rank=lora_rank,
        output_dir=f"{output_dir}/{method}",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    typer.echo(f"Loading GSM8K dataset...")
    train_ds = build_dataset(tokenizer, split="train")
    eval_ds = build_dataset(tokenizer, split="test")

    typer.echo(f"Starting {method.upper()} training on {model_id} for {steps} steps...")
    trainer = TRAINERS[method](config)
    metrics = trainer.train(train_ds, eval_dataset=eval_ds)

    typer.echo(f"\nTraining complete. Logs at {config.output_dir}/metrics.jsonl")
    if metrics:
        final = metrics[-1]
        typer.echo(f"Final step metrics: {final}")


@algo_app.command("eval")
def algo_eval(
    checkpoint: Annotated[str, typer.Argument(help="Path to saved model checkpoint")],
    dataset: Annotated[str, typer.Option()] = "gsm8k",
    max_samples: Annotated[int, typer.Option()] = 1319,
):
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .algorithms.evaluator import evaluate_gsm8k, evaluate_math

    typer.echo(f"Loading checkpoint from {checkpoint}...")
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    if dataset == "gsm8k":
        results = evaluate_gsm8k(model, tokenizer, max_samples=max_samples)
    elif dataset == "math":
        results = evaluate_math(model, tokenizer, max_samples=max_samples)
    else:
        typer.echo(f"Unknown dataset '{dataset}'. Choose from: gsm8k, math")
        raise typer.Exit(1)

    typer.echo(json.dumps(results, indent=2))
