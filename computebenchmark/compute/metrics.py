import time
from dataclasses import dataclass, field

import pynvml


@dataclass
class GPUSnapshot:
    timestamp: float
    vram_used_mb: float
    vram_total_mb: float
    gpu_util_pct: int
    mem_util_pct: int


@dataclass
class ThroughputResult:
    model_id: str
    phase: str  # "prefill" | "decode" | "train"
    batch_size: int
    seq_len: int
    tokens_per_second: float
    peak_vram_mb: float
    mean_gpu_util_pct: float
    iterations: int
    snapshots: list[GPUSnapshot] = field(default_factory=list)


class GPUMonitor:
    def __init__(self, device_index: int = 0):
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    def snapshot(self) -> GPUSnapshot:
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        return GPUSnapshot(
            timestamp=time.perf_counter(),
            vram_used_mb=mem.used / 1024**2,
            vram_total_mb=mem.total / 1024**2,
            gpu_util_pct=util.gpu,
            mem_util_pct=util.memory,
        )

    def __del__(self):
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
