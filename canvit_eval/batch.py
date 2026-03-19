"""Batch evaluation: reproduce ALL paper results in one command.

Covers: ADE20K segmentation, IN1K classification, ablation reconstruction.
Results go to task-specific subdirs: results/ade20k_seg/, results/in1k_clf/, results/recon/.
Each file includes a UTC timestamp in its filename.

Usage:
    # Sequential (crockett, single GPU):
    ADE20K_ROOT=... uv run python -m canvit_eval.batch --n-runs 1

    # Single task:
    ADE20K_ROOT=... uv run python -m canvit_eval.batch --tasks ade20k-seg --n-runs 5

    # SLURM array (Nibi — many short jobs >> 1 long one):
    uv run python -m canvit_eval.batch --dry-run --n-runs 5 > /tmp/eval_cmds.txt
    sbatch --array=1-$(wc -l < /tmp/eval_cmds.txt) eval.sbatch
    # In eval.sbatch: sed -n "${SLURM_ARRAY_TASK_ID}p" /tmp/eval_cmds.txt | bash
"""

import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

import tyro

from canvit_eval.policies import IN1K_POLICIES, PolicyName

log = logging.getLogger(__name__)


def _utc_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


# ── ADE20K segmentation constants ─────────────────────────────────────

ALL_POLICIES: list[str] = list(get_args(PolicyName))
DETERMINISTIC: set[str] = {"repeated_full_scene"}
ADE20K_RESOLUTIONS = [(512, 32, 32), (1024, 64, 8)]  # (scene_px, canvas_grid, batch_size)

DINOV3_VARIANTS: dict[str, str] = {
    "dv3b": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dv3s": "facebook/dinov3-vits16-pretrain-lvd1689m",
}
DINOV3_RESOLUTIONS = [128, 144, 160, 192, 256, 384, 512]

# CanViT single-glimpse probes (passive-vision comparison table).
CANVAS_GRIDS = [(512, 8), (512, 16), (512, 32), (1024, 64)]

# batch_size lookup by scene resolution (from ADE20K_RESOLUTIONS)
_BATCH_SIZE_BY_SCENE: dict[int, int] = {s: bs for s, _, bs in ADE20K_RESOLUTIONS}

# ── IN1K classification constants ──────────────────────────────────────
# IN1K_POLICIES imported from canvit_eval.policies (single source of truth).

# ── Ablation reconstruction constants ──────────────────────────────────
# Single source of truth for ablation HF repo IDs.
# Paper repo imports this via `from canvit_eval.batch import ABLATION_REPOS`.

ABLATION_REPOS: dict[str, str] = {
    "baseline":      "canvit/canvitb16-abl-baseline-2026-03-02",
    "qkvo-dcan256":  "canvit/canvitb16-abl-qkvo-dcan256-2026-03-02",
    "qkvo-dcan384":  "canvit/canvitb16-abl-qkvo-dcan384-2026-03-02",
    "dcan256":       "canvit/canvitb16-abl-dcan256-2026-03-02",
    "no-dense":      "canvit/canvitb16-abl-no-dense-2026-03-02",
    "no-fiid-1riid": "canvit/canvitb16-abl-no-fiid-1riid-2026-03-02",
    "no-fiid-2riid": "canvit/canvitb16-abl-no-fiid-2riid-2026-03-06",
    "no-bptt":       "canvit/canvitb16-abl-no-bptt-2026-03-06",
    "no-reads":      "canvit/canvitb16-abl-no-reads-2026-03-02",
    "no-vpe":        "canvit/canvitb16-abl-no-vpe-2026-03-03",
    "rw-stride6":    "canvit/canvitb16-abl-rw-stride6-2026-03-03",
    "vit-s":         "canvit/canvitb16-abl-vit-s-2026-03-03",
}


def _probe_repo(scene: int, grid: int) -> str:
    return f"canvit/probe-ade20k-40k-s{scene}-c{grid}-in21k"


# ── Job definition ─────────────────────────────────────────────────────

TaskName = Literal["ade20k-seg", "in1k-clf", "recon"]
ALL_TASKS: list[TaskName] = list(get_args(TaskName))


@dataclass
class EvalJob:
    """One evaluation to run."""
    task: TaskName
    args: list[str]
    output: Path


# ── Matrix builders ────────────────────────────────────────────────────


def _ade20k_seg_jobs(out_dir: Path, n_runs: int, n_timesteps: int, ts: str) -> list[EvalJob]:
    jobs: list[EvalJob] = []
    d = out_dir / "ade20k_seg"

    # CanViT multi-timestep policy evals
    for scene, grid, bs in ADE20K_RESOLUTIONS:
        probe = _probe_repo(scene, grid)
        for policy in ALL_POLICIES:
            n = 1 if policy in DETERMINISTIC else n_runs
            for run in range(n):
                out = d / f"{policy}_s{scene}_c{grid}_{ts}_r{run}.pt"
                jobs.append(EvalJob(
                    task="ade20k-seg",
                    args=["ade20k-seg", "--probe-repo", probe,
                          "--episode.policy", policy, "--episode.n-timesteps", str(n_timesteps),
                          "--episode.canvas-grid", str(grid), "--scene-size", str(scene),
                          "--batch-size", str(bs), "--output", str(out)],
                    output=out,
                ))

    # DINOv3 baseline probes (deterministic)
    for v, teacher_repo in DINOV3_VARIANTS.items():
        for res in DINOV3_RESOLUTIONS:
            out = d / f"{v}_{res}px_{ts}.pt"
            jobs.append(EvalJob(
                task="ade20k-seg",
                args=["ade20k-seg", "--model", "dinov3",
                      "--probe-repo", f"canvit/probe-ade20k-40k-{v}-{res}px",
                      "--teacher-repo", teacher_repo,
                      "--eval-resolution", str(res), "--output", str(out)],
                output=out,
            ))

    # CanViT single-glimpse probes (deterministic — full-scene viewpoint)
    for scene, grid in CANVAS_GRIDS:
        bs = _BATCH_SIZE_BY_SCENE.get(scene, 32)
        out = d / f"canvit_s{scene}_c{grid}_{ts}.pt"
        jobs.append(EvalJob(
            task="ade20k-seg",
            args=["ade20k-seg", "--probe-repo", _probe_repo(scene, grid),
                  "--episode.policy", "coarse_to_fine", "--episode.n-timesteps", "1",
                  "--episode.canvas-grid", str(grid), "--scene-size", str(scene),
                  "--batch-size", str(bs), "--output", str(out)],
            output=out,
        ))

    return jobs


def _in1k_clf_jobs(out_dir: Path, n_runs: int, n_timesteps: int, ts: str) -> list[EvalJob]:
    jobs: list[EvalJob] = []
    d = out_dir / "in1k_clf"
    for policy in IN1K_POLICIES:
        for run in range(n_runs):
            out = d / f"in1k_{policy}_{ts}_r{run}.pt"
            jobs.append(EvalJob(
                task="in1k-clf",
                args=["in1k-clf",
                      "--episode.policy", policy, "--episode.n-timesteps", str(n_timesteps),
                      "--output", str(out)],
                output=out,
            ))
    return jobs


def _recon_jobs(out_dir: Path, n_runs: int, ts: str) -> list[EvalJob]:
    jobs: list[EvalJob] = []
    d = out_dir / "recon"
    for slug, repo in ABLATION_REPOS.items():
        for run in range(n_runs):
            out = d / f"recon_{slug}_{ts}_r{run}.pt"
            jobs.append(EvalJob(
                task="recon",
                args=["reconstruction", "--model-repo", repo, "--output", str(out)],
                output=out,
            ))
    return jobs


def build_eval_matrix(
    out_dir: Path,
    n_runs: int,
    n_timesteps: int,
    tasks: list[TaskName],
) -> list[EvalJob]:
    """Generate the full list of eval jobs. Pure function, no side effects."""
    ts = _utc_timestamp()
    jobs: list[EvalJob] = []
    if "ade20k-seg" in tasks:
        jobs.extend(_ade20k_seg_jobs(out_dir, n_runs=n_runs, n_timesteps=n_timesteps, ts=ts))
    if "in1k-clf" in tasks:
        jobs.extend(_in1k_clf_jobs(out_dir, n_runs=n_runs, n_timesteps=n_timesteps, ts=ts))
    if "recon" in tasks:
        jobs.extend(_recon_jobs(out_dir, n_runs=n_runs, ts=ts))
    return jobs


# ── CLI ────────────────────────────────────────────────────────────────


@dataclass
class Args:
    out_dir: Path = Path("results")
    n_runs: int = 5
    n_timesteps: int = 21
    tasks: list[TaskName] = field(default_factory=lambda: list(ALL_TASKS))
    dry_run: bool = False


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    jobs = build_eval_matrix(
        out_dir=args.out_dir, n_runs=args.n_runs,
        n_timesteps=args.n_timesteps, tasks=args.tasks,
    )
    by_task = {}
    for j in jobs:
        by_task.setdefault(j.task, 0)
        by_task[j.task] += 1
    log.info("%d total eval jobs: %s", len(jobs), ", ".join(f"{k}={v}" for k, v in by_task.items()))

    if args.dry_run:
        for job in jobs:
            cmd = " ".join(["uv", "run", "python", "-m", "canvit_eval"] + job.args)
            print(cmd)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    failed = 0
    for i, job in enumerate(jobs):
        log.info("RUN  [%d/%d] %s", i + 1, len(jobs), job.output.name)
        result = subprocess.run([sys.executable, "-m", "canvit_eval"] + job.args)
        if result.returncode != 0:
            log.error("FAIL [%d/%d] %s (exit code %d)", i + 1, len(jobs), job.output.name, result.returncode)
            failed += 1
        else:
            done += 1

    log.info("DONE: %d completed, %d failed (of %d total)", done, failed, len(jobs))
    if failed:
        log.error("FAILED JOBS: %d — check logs above for details", failed)


if __name__ == "__main__":
    main(tyro.cli(Args))
