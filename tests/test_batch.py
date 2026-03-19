"""Tests for batch eval matrix generation — pure, no GPU/data needed."""

import re
from pathlib import Path

from canvit_eval.batch import (
    ABLATION_REPOS,
    ALL_POLICIES,
    ALL_TASKS,
    DETERMINISTIC,
    build_eval_matrix,
)

_TS_RE = re.compile(r"\d{8}T\d{6}Z")


def test_all_policies_from_literal():
    """ALL_POLICIES derived from PolicyName Literal, not hardcoded."""
    assert "coarse_to_fine" in ALL_POLICIES
    assert "entropy_coarse_to_fine" in ALL_POLICIES
    assert "repeated_full_scene" in ALL_POLICIES
    assert len(ALL_POLICIES) == 6


def test_deterministic_subset():
    assert DETERMINISTIC.issubset(set(ALL_POLICIES))


def test_ade20k_seg_n1():
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=["ade20k-seg"])
    # 6 policies × 2 res = 12 CanViT + 14 DINOv3 + 4 single-glimpse = 30
    assert len(jobs) == 30
    assert all(j.task == "ade20k-seg" for j in jobs)


def test_in1k_clf_n1():
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=["in1k-clf"])
    # 4 policies × 1 run = 4
    assert len(jobs) == 4
    assert all(j.task == "in1k-clf" for j in jobs)


def test_recon_n1():
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=["recon"])
    assert len(jobs) == len(ABLATION_REPOS)
    assert all(j.task == "recon" for j in jobs)


def test_all_tasks_combined():
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=ALL_TASKS)
    assert len(jobs) == 30 + 4 + len(ABLATION_REPOS)


def test_output_paths_unique():
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=5, n_timesteps=21, tasks=ALL_TASKS)
    paths = [j.output for j in jobs]
    assert len(paths) == len(set(paths)), "Duplicate output paths!"


def test_all_outputs_under_task_subdirs():
    out_dir = Path("/tmp/test_results")
    jobs = build_eval_matrix(out_dir, n_runs=1, n_timesteps=21, tasks=ALL_TASKS)
    expected_parents = {out_dir / "ade20k_seg", out_dir / "in1k_clf", out_dir / "recon"}
    actual_parents = {j.output.parent for j in jobs}
    assert actual_parents == expected_parents


def test_filenames_contain_timestamp():
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=ALL_TASKS)
    for j in jobs:
        assert _TS_RE.search(j.output.name), f"No timestamp in {j.output.name}"
