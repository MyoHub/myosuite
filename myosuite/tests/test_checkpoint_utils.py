# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for checkpoint discovery and loading (``find_checkpoint``, ``load_policy``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from myosuite.tests.support.sb3_models import raw_observations, vec_normalized_model
from myosuite.utils.checkpoint_utils import (
    find_checkpoint,
    find_vec_normalize,
    load_policy,
    resume_checkpoint,
)

pytestmark = pytest.mark.tier1


def test_find_checkpoint_returns_the_explicit_one_unchanged() -> None:
    """An explicit checkpoint short-circuits any search."""
    assert find_checkpoint(
        "myoElbowPose1D6MRandom-v0", checkpoint="some/model.pt"
    ) == Path("some/model.pt")


def test_find_checkpoint_prefers_a_local_logs_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A local ``logs/rsl_rl/<experiment>/<run>/model_*.pt`` wins over everything else."""
    # Pin the experiment name: its mjlab lookup is unavailable where mjlab is not installed (py3.14).
    monkeypatch.setattr(
        "myosuite.utils.checkpoint_utils.mjlab_experiment",
        lambda _env_id: "myo_elbow_pose",
    )
    run = tmp_path / "logs" / "rsl_rl" / "myo_elbow_pose" / "2026-01-01_00-00-00"
    run.mkdir(parents=True)
    (run / "model_0.pt").write_bytes(b"")
    (run / "model_200.pt").write_bytes(b"")

    found = find_checkpoint("myoElbowPose1D6MRandom-v0", roots=(tmp_path,))

    assert found == run / "model_200.pt"


def test_find_checkpoint_skips_runs_of_another_env_in_the_same_experiment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Env ids share an experiment name (the Exo twins have 7 actuators, not 6): only
    runs whose ``params/env.yaml`` records *env_id* are used."""
    monkeypatch.setattr(
        "myosuite.utils.checkpoint_utils.mjlab_experiment",
        lambda _env_id: "myo_elbow_pose",
    )
    experiment = tmp_path / "logs" / "rsl_rl" / "myo_elbow_pose"
    runs = {}
    for name, env_id in (
        ("2026-01-01_00-00-00", "myoElbowPose1D6MExoRandom-v0"),
        ("2026-02-01_00-00-00", "myoElbowPose1D6MRandom-v0"),  # newer, other env
    ):
        runs[env_id] = experiment / name
        (runs[env_id] / "params").mkdir(parents=True)
        # Shape of the dump: the env id sits in the robot entity's CpuTaskSpec.
        (runs[env_id] / "params" / "env.yaml").write_text(
            "scene:\n  entities:\n    robot:\n"
            "      spec_fn: !!python/object/apply:functools.partial\n"
            "        state: !!python/tuple\n        - !!python/tuple\n"
            "          - !!python/object:cpu_reference.CpuTaskSpec\n"
            f"            env_id: {env_id}\n"
            "viewer:\n  env_idx: 0\n"
        )
        (runs[env_id] / "model_100.pt").write_bytes(b"")
    baseline = tmp_path / "baselines" / "checkpoints" / "myoElbowPose1D6MFixed-v0"
    baseline.mkdir(parents=True)
    (baseline / "model_9.pt").write_bytes(b"")

    for env_id, run in runs.items():
        assert find_checkpoint(env_id, roots=(tmp_path,)) == run / "model_100.pt"
    # No run of its own: the env's baseline, not a sibling's run.
    assert (
        find_checkpoint("myoElbowPose1D6MFixed-v0", roots=(tmp_path,))
        == baseline / "model_9.pt"
    )


def test_find_checkpoint_reads_the_env_id_train_mjlab_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Runs are told apart by the ``params/env.yaml`` that ``train_mjlab.py`` writes."""
    pytest.importorskip("mjlab")
    from dataclasses import asdict

    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.os import dump_yaml

    import myosuite.envs.myo.backends.mjlab  # noqa: F401  (registers the twins)

    monkeypatch.setattr(
        "myosuite.core.hf_io.download_baseline_checkpoint", lambda env_id: None
    )
    experiment = tmp_path / "logs" / "rsl_rl" / "myo_elbow_pose"
    runs = {}
    for name, env_id in (
        ("2026-01-01_00-00-00", "myoElbowPose1D6MExoRandom-v0"),
        ("2026-02-01_00-00-00", "myoElbowPose1D6MRandom-v0"),
    ):
        runs[env_id] = experiment / name
        dump_yaml(runs[env_id] / "params" / "env.yaml", asdict(load_env_cfg(env_id)))
        (runs[env_id] / "model_100.pt").write_bytes(b"")

    for env_id, run in runs.items():
        assert find_checkpoint(env_id, roots=(tmp_path,)) == run / "model_100.pt"
    assert find_checkpoint("myoElbowPose1D6MExoFixed-v0", roots=(tmp_path,)) is None


def _run(experiment: Path, name: str, env_id: str | None, *checkpoints: str) -> Path:
    """A ``logs/rsl_rl/<experiment>/<name>`` run trained on *env_id* (``None``: legacy
    run without ``params/env.yaml``)."""
    run = experiment / name
    (run / "params").mkdir(parents=True)
    if env_id is not None:
        (run / "params" / "env.yaml").write_text(f"env_id: {env_id}\n")
    for checkpoint in checkpoints:
        (run / checkpoint).write_bytes(b"")
    return run


def test_resume_checkpoint_skips_runs_of_another_env_in_the_same_experiment(
    tmp_path: Path,
) -> None:
    """``--agent.resume`` continues this task's newest run, not the experiment's: mjlab's
    own lookup would resume the 6-actuator run for the 7-actuator Exo twin."""
    pytest.importorskip("mjlab")
    from mjlab.utils.os import get_checkpoint_path

    experiment = tmp_path / "logs" / "rsl_rl" / "myo_elbow_pose"
    exo = _run(
        experiment,
        "2026-09-20_10-00-00",
        "myoElbowPose1D6MExoRandom-v0",
        "model_500.pt",
    )
    plain = _run(
        experiment, "2026-09-21_10-00-00", "myoElbowPose1D6MRandom-v0", "model_500.pt"
    )
    assert get_checkpoint_path(experiment, ".*", "model_.*.pt").parent == plain

    for env_id, run in (
        ("myoElbowPose1D6MExoRandom-v0", exo),
        ("myoElbowPose1D6MRandom-v0", plain),
    ):
        assert resume_checkpoint(experiment, env_id) == run / "model_500.pt"
    with pytest.raises(ValueError, match="myoElbowPose1D6MExoFixed-v0"):
        resume_checkpoint(experiment, "myoElbowPose1D6MExoFixed-v0")


def test_resume_checkpoint_follows_mjlab_run_and_checkpoint_selection(
    tmp_path: Path,
) -> None:
    """Newest qualifying run, newest checkpoint; an exact ``--agent.load-run`` is kept."""
    pytest.importorskip("mjlab")
    env_id = "myoElbowPose1D6MRandom-v0"
    experiment = tmp_path / "logs" / "rsl_rl" / "myo_elbow_pose"
    legacy = _run(experiment, "2026-01-01_00-00-00", None, "model_900.pt")
    old = _run(experiment, "2026-02-01_00-00-00", env_id, "model_50.pt", "model_100.pt")
    other = _run(
        experiment, "2026-03-01_00-00-00", "myoElbowPose1D6MFixed-v0", "model_7.pt"
    )
    _run(experiment, "2026-04-01_00-00-00", env_id)  # crashed before its first save
    _run(experiment, "wandb_checkpoints", None, "model_1.pt")

    assert resume_checkpoint(experiment, env_id) == old / "model_100.pt"
    assert resume_checkpoint(experiment, env_id, "2026-01.*") == legacy / "model_900.pt"
    assert (
        resume_checkpoint(experiment, env_id, load_checkpoint="model_50.pt")
        == old / "model_50.pt"
    )
    # Naming another env's run exactly is an explicit warm start (same action space).
    assert resume_checkpoint(experiment, env_id, other.name) == other / "model_7.pt"
    with pytest.raises(ValueError, match="No run"):
        resume_checkpoint(tmp_path / "missing", env_id)


def test_find_checkpoint_falls_back_to_a_local_baseline(tmp_path: Path) -> None:
    """With no local run, the repository's default baseline checkpoint is used."""
    baseline = tmp_path / "baselines" / "checkpoints" / "myoElbowPose1D6MRandom-v0"
    baseline.mkdir(parents=True)
    (baseline / "model_200.pt").write_bytes(b"")

    found = find_checkpoint("myoElbowPose1D6MRandom-v0", roots=(tmp_path,))

    assert found == baseline / "model_200.pt"


def test_find_checkpoint_falls_back_to_the_hf_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No local baseline (e.g. a fresh clone) downloads it from Hugging Face instead."""
    hf_dir = tmp_path / "hf_cache" / "myoElbowPose1D6MRandom-v0"
    hf_dir.mkdir(parents=True)
    (hf_dir / "model_200.pt").write_bytes(b"")

    monkeypatch.setattr(
        "myosuite.core.hf_io.download_baseline_checkpoint",
        lambda env_id: hf_dir if env_id == "myoElbowPose1D6MRandom-v0" else None,
    )

    found = find_checkpoint("myoElbowPose1D6MRandom-v0", roots=(tmp_path,))

    assert found == hf_dir / "model_200.pt"


def test_find_checkpoint_falls_back_to_sb3_zip_when_no_hf_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An env with no baseline anywhere (local or Hugging Face) still tries the SB3 zip."""
    (tmp_path / "policy.zip").write_bytes(b"")
    monkeypatch.setattr(
        "myosuite.core.hf_io.download_baseline_checkpoint", lambda env_id: None
    )

    found = find_checkpoint(
        "myoElbowPose1D6MRandom-v0", roots=(tmp_path,), sb3_zip="policy.zip"
    )

    assert found == tmp_path / "policy.zip"


def test_find_checkpoint_returns_none_when_nothing_is_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No local run, no local baseline, no Hugging Face baseline, no SB3 zip: None."""
    monkeypatch.setattr(
        "myosuite.core.hf_io.download_baseline_checkpoint", lambda env_id: None
    )

    assert find_checkpoint("myoElbowPose1D6MRandom-v0", roots=(tmp_path,)) is None


def test_find_vec_normalize_prefers_the_checkpoint_specific_file(
    tmp_path: Path,
) -> None:
    """``<stem>_vecnormalize.pkl`` wins over RL Zoo's and the legacy MyoSuite name."""
    checkpoint = tmp_path / "ppo_final.zip"
    assert find_vec_normalize(checkpoint) is None
    for name in ("vec_normalize.pkl", "vecnormalize.pkl", "ppo_final_vecnormalize.pkl"):
        (tmp_path / name).write_bytes(b"")
        assert find_vec_normalize(checkpoint) == tmp_path / name


@pytest.mark.parametrize("algo_name", ["PPO", "SAC"])
def test_load_policy_normalizes_observations_like_training(
    algo_name: str, tmp_path: Path
) -> None:
    """An SB3 policy (PPO or SAC) acts on raw observations through its VecNormalize."""
    model, venv = vec_normalized_model(algo_name)
    try:
        raw = raw_observations(venv)
        expected, _ = model.predict(venv.normalize_obs(raw), deterministic=True)
        unnormalized, _ = model.predict(raw, deterministic=True)
        model.save(tmp_path / "policy.zip")
        venv.save(tmp_path / "vecnormalize.pkl")

        policy = load_policy(venv.envs[0], tmp_path / "policy.zip")
    finally:
        venv.close()

    assert np.abs(expected - unnormalized).max() > 1e-2
    np.testing.assert_allclose(policy(raw), expected, atol=1e-6)
