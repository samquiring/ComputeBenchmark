import csv
import json
from dataclasses import asdict
from pathlib import Path

from .metrics import ThroughputResult


def _to_row(result: ThroughputResult) -> dict:
    d = asdict(result)
    d.pop("snapshots")
    return d


def save_json(results: list[ThroughputResult], path: str | Path) -> None:
    Path(path).write_text(json.dumps([_to_row(r) for r in results], indent=2))


def save_csv(results: list[ThroughputResult], path: str | Path) -> None:
    rows = [_to_row(r) for r in results]
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def print_table(results: list[ThroughputResult]) -> None:
    cols = f"{'Model':<40} {'Phase':<8} {'BS':>3} {'SeqLen':>7} {'tok/s':>10} {'VRAM MB':>9} {'GPU%':>5}"
    print(cols)
    print("-" * len(cols))
    for r in results:
        print(
            f"{r.model_id:<40} {r.phase:<8} {r.batch_size:>3} {r.seq_len:>7} "
            f"{r.tokens_per_second:>10.1f} {r.peak_vram_mb:>9.1f} {r.mean_gpu_util_pct:>5.1f}"
        )
