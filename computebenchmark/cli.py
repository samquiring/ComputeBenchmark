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
    from .algorithms.trainers import TRAINERS, get_trainer
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
    trainer = get_trainer(method)(config)
    metrics = trainer.train(train_ds, eval_dataset=eval_ds)

    typer.echo(f"\nTraining complete. Logs at {config.output_dir}/metrics.jsonl")
    if metrics:
        final = metrics[-1]
        typer.echo(f"Final step metrics: {final}")


@algo_app.command("race")
def algo_race(
    model_id: Annotated[str, typer.Option("--model-id", "-m")],
    baseline_steps: Annotated[int, typer.Option(help="Steps to run PPO baseline")] = 500,
    max_steps: Annotated[int, typer.Option(help="Step budget for challenger methods")] = 500,
    eval_every: Annotated[int, typer.Option()] = 25,
    output_dir: Annotated[str, typer.Option("--output-dir", "-o")] = "/workspace/results/race",
):
    import json
    from dataclasses import asdict
    from transformers import AutoTokenizer
    from .algorithms.trainers import TRAINERS, get_trainer
    from .algorithms.trainers.base import TrainingConfig
    from .data.gsm8k import build_dataset

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    typer.echo("Loading GSM8K...")
    train_ds = build_dataset(tokenizer, split="train")
    eval_ds = build_dataset(tokenizer, split="test")

    # ── Phase 1: run PPO baseline ─────────────────────────────────────────────
    typer.echo(f"\n{'='*60}")
    typer.echo(f"Phase 1: REINFORCE baseline for {baseline_steps} steps")
    typer.echo(f"{'='*60}")

    ppo_config = TrainingConfig(
        model_id=model_id,
        num_steps=baseline_steps,
        eval_every=eval_every,
        output_dir=f"{output_dir}/ppo",
    )
    _, ppo_convergence = get_trainer("ppo")(ppo_config).train(train_ds, eval_ds)
    target_accuracy = ppo_convergence.final_accuracy
    typer.echo(f"\nREINFORCE final accuracy: {target_accuracy:.3f} — this is the target.")

    # ── Phase 2: challengers race to match PPO ────────────────────────────────
    all_results = {"ppo": asdict(ppo_convergence)}

    for method in ["grpo", "dapo", "rlvr"]:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"Phase 2: {method.upper()} racing to {target_accuracy:.3f}")
        typer.echo(f"{'='*60}")

        config = TrainingConfig(
            model_id=model_id,
            num_steps=max_steps,
            eval_every=eval_every,
            output_dir=f"{output_dir}/{method}",
            target_accuracy=target_accuracy,
        )
        _, convergence = get_trainer(method)(config).train(train_ds, eval_ds)
        all_results[method] = asdict(convergence)

    # ── Summary table ─────────────────────────────────────────────────────────
    typer.echo(f"\n{'='*60}")
    typer.echo("RESULTS SUMMARY")
    typer.echo(f"PPO baseline: {baseline_steps} steps → {target_accuracy:.3f} accuracy\n")

    header = f"{'Method':<8}  {'Reached':>7}  {'Steps':>6}  {'Wall clock':>12}  {'Tokens seen':>14}  {'Final acc':>9}"
    typer.echo(header)
    typer.echo("-" * len(header))

    for method, r in all_results.items():
        reached = "YES" if r["reached_target"] else "NO"
        steps = str(r["steps_to_target"]) if r["steps_to_target"] is not None else f">{max_steps}"
        wall = f"{r['wall_clock_seconds_to_target']/60:.1f}min" if r["wall_clock_seconds_to_target"] else "—"
        tokens = f"{r['tokens_seen_to_target']:,}" if r["tokens_seen_to_target"] else "—"
        typer.echo(f"{method:<8}  {reached:>7}  {steps:>6}  {wall:>12}  {tokens:>14}  {r['final_accuracy']:>9.3f}")

    results_path = f"{output_dir}/race_summary.json"
    import os; os.makedirs(output_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    typer.echo(f"\nFull results saved to {results_path}")


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
