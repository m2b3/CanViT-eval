"""Tests for viewing policies."""

import pytest
import torch
from canvit.policies import level_viewpoints

from canvit_eval.policies import (
    StaticPolicy,
    fine_to_coarse_viewpoints,
    make_policy,
)


def test_level_viewpoints_count() -> None:
    """Tests the core canvit.policies.level_viewpoints (imported by eval)."""
    assert len(level_viewpoints(0)) == 1    # full scene
    assert len(level_viewpoints(1)) == 4    # 2×2
    assert len(level_viewpoints(2)) == 16   # 4×4


def test_level0_is_full_scene() -> None:
    vps = level_viewpoints(0)
    y, x, s = vps[0]
    assert (y, x, s) == (0.0, 0.0, 1.0)


def test_c2f_starts_with_full_scene() -> None:
    policy = make_policy("coarse_to_fine", batch_size=2, device=torch.device("cpu"), n_viewpoints=5)
    vp0 = policy.step(0, None)  # type: ignore[arg-type]
    assert vp0.scales[0].item() == 1.0
    assert vp0.centers[0, 0].item() == 0.0
    assert vp0.centers[0, 1].item() == 0.0


def test_f2c_starts_with_finest() -> None:
    vps = fine_to_coarse_viewpoints(batch_size=1, device=torch.device("cpu"), n_viewpoints=5)
    # First viewpoint should be at the finest scale (smallest)
    assert vps[0].scales[0].item() < 1.0


def test_repeated_full_scene_same_every_step() -> None:
    policy = make_policy("repeated_full_scene", batch_size=2, device=torch.device("cpu"), n_viewpoints=5)
    for t in range(5):
        vp = policy.step(t, None)  # type: ignore[arg-type]
        assert vp.scales[0].item() == 1.0


def test_all_policies_produce_correct_count() -> None:
    T = 5
    for name in ["coarse_to_fine", "fine_to_coarse", "random", "full_then_random", "repeated_full_scene"]:
        policy = make_policy(name, batch_size=2, device=torch.device("cpu"), n_viewpoints=T)  # type: ignore[arg-type]
        assert isinstance(policy, StaticPolicy)
        assert len(policy._viewpoints) == T, f"{name}: expected {T} viewpoints, got {len(policy._viewpoints)}"


def test_unknown_policy_raises() -> None:
    with pytest.raises(ValueError):
        make_policy("nonexistent", batch_size=1, device=torch.device("cpu"), n_viewpoints=1)  # type: ignore[arg-type]
