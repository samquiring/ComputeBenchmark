import time
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import GPUMonitor, ThroughputResult

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class ComputeConfig:
    model_id: str
    batch_sizes: list[int] = field(default_factory=lambda: [1, 4, 8])
    prompt_lengths: list[int] = field(default_factory=lambda: [128, 512, 1024])
    generation_length: int = 256
    warmup_iters: int = 3
    bench_iters: int = 10
    dtype: str = "bfloat16"


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure_prefill(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    monitor: GPUMonitor,
    iterations: int,
) -> tuple[float, float, list]:
    snapshots = []
    _sync()
    t0 = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            model(input_ids)
        _sync()
        snapshots.append(monitor.snapshot())
    elapsed = time.perf_counter() - t0
    tps = (input_ids.numel() * iterations) / elapsed
    peak_vram = max(s.vram_used_mb for s in snapshots)
    return tps, peak_vram, snapshots


def _measure_decode(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    monitor: GPUMonitor,
    iterations: int,
    gen_len: int = 256,
) -> tuple[float, float, list]:
    snapshots = []
    _sync()
    t0 = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            model.generate(
                input_ids,
                max_new_tokens=gen_len,
                do_sample=False,
                pad_token_id=model.config.eos_token_id,
            )
        _sync()
        snapshots.append(monitor.snapshot())
    elapsed = time.perf_counter() - t0
    tps = (gen_len * input_ids.shape[0] * iterations) / elapsed
    peak_vram = max(s.vram_used_mb for s in snapshots)
    return tps, peak_vram, snapshots


def _measure_train(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    monitor: GPUMonitor,
    iterations: int,
) -> tuple[float, float, list]:
    # Forward + backward without optimizer step to isolate gradient compute cost
    snapshots = []
    _sync()
    t0 = time.perf_counter()
    for _ in range(iterations):
        labels = input_ids.clone()
        out = model(input_ids, labels=labels)
        out.loss.backward()
        model.zero_grad()
        _sync()
        snapshots.append(monitor.snapshot())
    elapsed = time.perf_counter() - t0
    tps = (input_ids.numel() * iterations) / elapsed
    peak_vram = max(s.vram_used_mb for s in snapshots)
    return tps, peak_vram, snapshots


def run(config: ComputeConfig) -> list[ThroughputResult]:
    dtype = DTYPE_MAP[config.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, torch_dtype=dtype, device_map="auto"
    )
    model.eval()
    monitor = GPUMonitor()
    results = []

    for batch_size in config.batch_sizes:
        for seq_len in config.prompt_lengths:
            input_ids = torch.randint(100, 50000, (batch_size, seq_len), device="cuda")

            for _ in range(config.warmup_iters):
                with torch.no_grad():
                    model(input_ids)
            _sync()

            for phase, fn, kwargs in [
                ("prefill", _measure_prefill, {}),
                ("decode", _measure_decode, {"gen_len": config.generation_length}),
            ]:
                tps, peak_vram, snaps = fn(
                    model, input_ids, monitor, config.bench_iters, **kwargs
                )
                mean_util = sum(s.gpu_util_pct for s in snaps) / len(snaps)
                results.append(
                    ThroughputResult(
                        model_id=config.model_id,
                        phase=phase,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        tokens_per_second=tps,
                        peak_vram_mb=peak_vram,
                        mean_gpu_util_pct=mean_util,
                        iterations=config.bench_iters,
                        snapshots=snaps,
                    )
                )

            # training throughput requires grad
            model.train()
            tps, peak_vram, snaps = _measure_train(
                model, input_ids, monitor, config.bench_iters
            )
            mean_util = sum(s.gpu_util_pct for s in snaps) / len(snaps)
            results.append(
                ThroughputResult(
                    model_id=config.model_id,
                    phase="train",
                    batch_size=batch_size,
                    seq_len=seq_len,
                    tokens_per_second=tps,
                    peak_vram_mb=peak_vram,
                    mean_gpu_util_pct=mean_util,
                    iterations=config.bench_iters,
                    snapshots=snaps,
                )
            )
            model.eval()

    return results
