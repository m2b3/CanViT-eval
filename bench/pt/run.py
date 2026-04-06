"""Real-time latency benchmark: CanViT vs DINOv3.

Measures per-forward-pass LATENCY (batch_size=1, full GPU sync before/after
each iteration). This is the relevant metric for real-time inference
(webcam, robotics, etc.) — NOT throughput.

Key concept: "scene resolution" determines workload for BOTH models.
  - DINOv3: input_px = scene_px → (scene_px/16)² patches.
  - CanViT: glimpse always 128px (64 patches), canvas_grid = scene_px/16.
            Canvas spatial token count matches DINOv3's patch count.

Methodology:
  1. Warmup phase: N iterations to trigger torch.compile and prime caches.
  2. Peak memory measurement (after warmup).
  3. Measurement phase: timed iterations with per-iter GPU sync.
     Budget and iter count are for measurement only (exclude warmup).

Results streamed to JSONL (survives OOM/crashes).
One invocation = one (model, scene_px, dtype, compiled) config.

Usage:
    uv run python bench/pt/run.py --model canvit --device cuda --scene-px 512 --compiled --dtype amp-bf16
    uv run python bench/pt/run.py --model dinov3-vitb16 --device cpu --scene-px 512 --dtype fp32 --num-threads 1
"""

import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Literal

import torch
import torch._inductor.config
import tyro

from canvit_pytorch.backbone import create_backbone
from canvit_pytorch.model.base import CanViT, CanViTConfig
from canvit_pytorch.viewpoint import Viewpoint, sample_at_viewpoint
from canvit_utils.teacher import load_teacher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("bench")

DINOV3_REPOS = {
    "dinov3-vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3-vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
}
CANVIT_GLIMPSE_PX = 128
RESULTS_DIR = Path(__file__).parent / "results"


@dataclass(frozen=True)
class Args:
    model: Literal["canvit", "dinov3-vitb16", "dinov3-vits16"] = "canvit"
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    scene_px: int = 512
    """Scene resolution in pixels. Teacher gets this as input_px.
    CanViT gets canvas_grid = scene_px // 16, glimpse fixed at 128px."""
    compiled: bool = False
    combo_kernels: bool = False
    """Enable torch._inductor.config.combo_kernels (requires --compiled)."""
    dtype: Literal["fp32", "amp-bf16"] = "amp-bf16"
    batch_size: int = 1
    time_budget_s: float = 120.0
    """Measurement time budget in seconds (excludes model loading)."""
    max_iters: int = 100
    """Stop after this many iterations even if time budget not exhausted."""
    num_threads: int = 0
    """Number of CPU threads (0 = PyTorch default). Only relevant for --device cpu."""
    warmup_iters: int = 3
    """Warmup iterations before measurement (iter 0 triggers torch.compile)."""


def _weight_dtype(args: Args) -> torch.dtype:
    # Always fp32: AMP autocast handles bf16 compute, weights stay fp32
    return torch.float32


@contextmanager
def _autocast(args: Args, device: torch.device) -> Iterator[None]:
    if args.dtype == "amp-bf16":
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            yield
    else:
        yield


def _run_id(args: Args, ts: str) -> str:
    c = "c" if args.compiled else "e"
    combo = "_combo" if args.combo_kernels else ""
    bs = f"_bs{args.batch_size}" if args.batch_size != 1 else ""
    dev = f"_{args.device}" if args.device != "cuda" else ""
    thr = f"_t{args.num_threads}" if args.num_threads > 0 else ""
    if args.model == "canvit":
        cg = args.scene_px // 16
        return f"canvit_{c}_{args.dtype}_{args.scene_px}px_cg{cg}{bs}{combo}{dev}{thr}_{ts}"
    return f"{args.model}_{c}_{args.dtype}_{args.scene_px}px{bs}{combo}{dev}{thr}_{ts}"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _measure_peak_mb(device: torch.device, fn: Callable[[], None]) -> float | None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        _sync(device)
        fn()
        _sync(device)
        return round(torch.cuda.max_memory_allocated() / 1e6, 1)
    if device.type == "mps":
        torch.mps.empty_cache()
        _sync(device)
        before = torch.mps.current_allocated_memory()
        fn()
        _sync(device)
        return round((torch.mps.current_allocated_memory() - before) / 1e6, 1)
    return None


def _measure_streaming(
    fn: Callable[[], None],
    out_path: Path,
    meta: dict,
    time_budget_s: float,
    max_iters: int,
    warmup_iters: int,
    device: torch.device,
) -> None:
    """Warmup (compilation), then measure fn repeatedly, streaming to JSONL.

    Phase 1: N_WARMUP warmup iterations (not timed against budget).
             Iter 0 triggers torch.compile. Peak memory measured after warmup.
    Phase 2: Timed iterations until time_budget_s or max_iters (whichever first).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(json.dumps({"type": "meta", **meta}) + "\n")
        f.flush()

        # -- Warmup phase (not counted toward budget) --
        for w in range(warmup_iters):
            _sync(device)
            t0 = time.perf_counter()
            fn()
            _sync(device)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            tag = "cold/compile" if w == 0 else "warmup"
            log.info("  warmup %d (%s): %.1fms", w, tag, elapsed_ms)
            row = {"type": "warmup", "i": w, "ms": round(elapsed_ms, 4)}
            f.write(json.dumps(row) + "\n")
            f.flush()

        # -- Peak memory (after warmup, before measurement) --
        peak_mb = _measure_peak_mb(device, fn)
        if peak_mb is not None:
            log.info("  peak memory: %.1f MB", peak_mb)
            f.write(json.dumps({"type": "peak_mem", "peak_mem_mb": peak_mb}) + "\n")
            f.flush()

        # -- Measurement phase (budget starts here) --
        i = 0
        wall_start = time.perf_counter()
        while True:
            _sync(device)
            t0 = time.perf_counter()
            fn()
            _sync(device)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            wall_s = time.perf_counter() - wall_start

            row = {"type": "iter", "i": i, "ms": round(elapsed_ms, 4), "wall_s": round(wall_s, 3)}
            f.write(json.dumps(row) + "\n")
            f.flush()

            if i <= 3 or i % 50 == 0:
                log.info("  iter %d: %.2fms (wall %.1fs)", i, elapsed_ms, wall_s)

            i += 1
            if wall_s >= time_budget_s or i >= max_iters:
                break

    log.info("  %d measured iterations in %.1fs -> %s", i, time.perf_counter() - wall_start, out_path)


def _build_dinov3(args: Args, device: torch.device) -> Callable[[], None]:
    repo = DINOV3_REPOS[args.model]
    log.info("Loading %s from %s...", args.model, repo)
    t0 = time.perf_counter()
    teacher = load_teacher(repo, device).to(dtype=_weight_dtype(args)).eval()
    n_params = sum(p.numel() for p in teacher.parameters())
    log.info("  loaded in %.1fs, %.1fM params", time.perf_counter() - t0, n_params / 1e6)

    if args.compiled:
        log.info("  torch.compile...")
        t0 = time.perf_counter()
        teacher.model = torch.compile(teacher.model)
        log.info("  registered in %.1fs", time.perf_counter() - t0)

    x = torch.randn(args.batch_size, 3, args.scene_px, args.scene_px,
                     device=device, dtype=_weight_dtype(args))
    n_patches = (args.scene_px // 16) ** 2
    log.info("  input: %dpx -> %d patches", args.scene_px, n_patches)

    def fwd(x=x) -> None:
        teacher.forward_norm_features(x)

    return fwd


def _build_canvit(args: Args, device: torch.device) -> Callable[[], None]:
    log.info("Creating CanViT...")
    backbone = create_backbone("vitb16")
    model = CanViT(backbone=backbone, cfg=CanViTConfig())
    model = model.to(device=device, dtype=_weight_dtype(args)).eval()
    n_params = sum(p.numel() for p in model.parameters())
    log.info("  %.1fM params", n_params / 1e6)

    if args.compiled:
        log.info("  torch.compile (fullgraph)...")
        t0 = time.perf_counter()
        model.compile()
        log.info("  registered in %.1fs", time.perf_counter() - t0)

    canvas_grid = args.scene_px // 16
    bs = args.batch_size
    image = torch.randn(bs, 3, CANVIT_GLIMPSE_PX, CANVIT_GLIMPSE_PX,
                         device=device, dtype=_weight_dtype(args))
    vp = Viewpoint.full_scene(batch_size=bs, device=device)
    glimpse = sample_at_viewpoint(spatial=image, viewpoint=vp, glimpse_size_px=CANVIT_GLIMPSE_PX)
    log.info("  glimpse: %dpx (64 patches), canvas: %dx%d (%d tokens)",
             CANVIT_GLIMPSE_PX, canvas_grid, canvas_grid, canvas_grid ** 2)

    def run(glimpse=glimpse, vp=vp) -> None:
        state = model.init_state(batch_size=bs, canvas_grid_size=canvas_grid)
        model(glimpse=glimpse, state=state, viewpoint=vp)

    return run


def _device_info(device: torch.device) -> tuple[str, float | None]:
    if device.type == "cuda":
        return torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1e9
    if device.type == "mps":
        return "Apple Silicon (MPS)", None
    # XXX: Linux-only. On macOS, returns "CPU" — this is why hw_bench_table.typ
    # hardcodes the CPU model name. Fix: also try `sysctl -n machdep.cpu.brand_string`
    # on macOS, or `platform.processor()` as a portable fallback.
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip(), None
    except FileNotFoundError:
        pass
    return "CPU", None


def main() -> None:
    args = tyro.cli(Args)

    assert args.scene_px % 16 == 0, f"scene_px must be divisible by 16, got {args.scene_px}"
    if args.combo_kernels:
        assert args.compiled, "--combo-kernels requires --compiled"
        torch._inductor.config.combo_kernels = True
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        log.info("Set num_threads = %d", args.num_threads)

    device = torch.device(args.device)
    dev_name, dev_mem_gb = _device_info(device)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rid = _run_id(args, ts)

    log.info("Device: %s%s", dev_name, f" ({dev_mem_gb:.1f} GB)" if dev_mem_gb else "")
    log.info("torch: %s", torch.__version__)
    log.info("Run: %s", rid)
    log.info("Args: %s", args)

    builder = _build_canvit if args.model == "canvit" else _build_dinov3
    with torch.inference_mode():
        fwd = builder(args, device)

        meta = {
            "device_name": dev_name,
            "torch_version": torch.__version__,
            "num_threads_actual": torch.get_num_threads(),
            "run_id": rid,
            "timestamp": ts,
            **{k: v for k, v in args.__dict__.items()},
        }
        if dev_mem_gb is not None:
            meta["device_mem_gb"] = round(dev_mem_gb, 1)
        if args.model == "canvit":
            meta["canvas_grid"] = args.scene_px // 16
            meta["glimpse_px"] = CANVIT_GLIMPSE_PX

        out_path = RESULTS_DIR / f"bench_{rid}.jsonl"
        log.info("Measuring for %.0fs...", args.time_budget_s)
        with _autocast(args, device):
            _measure_streaming(fwd, out_path, meta, args.time_budget_s, args.max_iters, args.warmup_iters, device)

    log.info("Done.")


if __name__ == "__main__":
    main()
