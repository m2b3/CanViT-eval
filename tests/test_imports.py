"""Smoke tests: verify public APIs are importable and structurally sound."""

from typing import get_args

import torch

from canvit_eval.policies import IN1K_POLICIES, PolicyName, make_policy


def test_config_repos_are_hf_style():
    from canvit_eval.config import DEFAULT_MODEL_REPO, TEACHER_REPO, EpisodeConfig

    assert "/" in DEFAULT_MODEL_REPO
    assert "/" in TEACHER_REPO
    cfg = EpisodeConfig()
    assert cfg.n_timesteps > 0


def test_episode_step_has_expected_fields():
    import dataclasses

    from canvit_eval.episode import EpisodeStep

    fields = {f.name for f in dataclasses.fields(EpisodeStep)}
    assert {"t", "state", "output", "viewpoint"} == fields


def test_static_policies_constructible():
    """All non-entropy policies can be constructed and produce valid viewpoints."""
    static_names = [n for n in get_args(PolicyName) if n != "entropy_coarse_to_fine"]
    for name in static_names:
        policy = make_policy(name, batch_size=2, device=torch.device("cpu"), n_viewpoints=5)
        assert hasattr(policy, "step")
        vp = policy.step(t=0, state=None)  # type: ignore[arg-type]  # StaticPolicy ignores state
        assert vp.centers.shape == (2, 2)
        assert vp.scales.shape == (2,)
        assert (vp.scales > 0).all()
        assert (vp.scales <= 1).all()


def test_in1k_policies_are_subset():
    all_policies = set(get_args(PolicyName))
    assert set(IN1K_POLICIES).issubset(all_policies)
    assert "entropy_coarse_to_fine" not in IN1K_POLICIES


def test_entropy_policy_requires_probe():
    import pytest

    with pytest.raises(AssertionError):
        make_policy("entropy_coarse_to_fine", batch_size=1, device=torch.device("cpu"), n_viewpoints=5)


def test_batch_constants():
    from canvit_eval.batch import ALL_POLICIES, ALL_TASKS, DETERMINISTIC, _BATCH_SIZE_BY_SCENE

    assert set(ALL_POLICIES) == set(get_args(PolicyName))
    assert DETERMINISTIC.issubset(set(ALL_POLICIES))
    assert len(ALL_TASKS) >= 2
    # OOM fix: 1024px scenes must have reduced batch size
    assert _BATCH_SIZE_BY_SCENE[1024] < _BATCH_SIZE_BY_SCENE[512]


def test_evaluate_protocol_shape():
    from canvit_eval.evaluate import MetricAccumulator

    assert hasattr(MetricAccumulator, "update")
    assert hasattr(MetricAccumulator, "compute")


def test_task_modules_importable():
    from canvit_eval.tasks import ade20k_seg_mIoU, in1k_clf, reconstruction  # noqa: F401
