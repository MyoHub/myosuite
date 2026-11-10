"""DynSyn trainer: resume rules, checkpoints that carry their map, a short run played back."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mjlab")

import torch  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402

import myosuite.envs.myo.backends.mjlab  # noqa: E402, F401  (registers the twins)
from myosuite.envs.myo.backends.mjlab.tasks.mdp.action_map import (  # noqa: E402
    action_map_term,
    apply_action_map,
)
from myosuite.utils.action_map import (  # noqa: E402
    DynSyn,
    DynSynGroups,
    trained_action_map,
)
from myosuite.utils.rslrl_policy import load_rslrl_policy  # noqa: E402
from myosuite_dynsyn.train import (  # noqa: E402
    DynSynTrainConfig,
    resolve_dynsyn,
    track_action_map_in_checkpoints,
    train,
)

pytestmark = pytest.mark.tier2

TASK = "myoElbowPose1D6MRandom-v0"
GROUPS = [[0, 1, 2], [3, 4, 5]]


def _cfg(num_envs: int):
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = num_envs
    return cfg


@pytest.fixture(scope="module")
def groups_file(tmp_path_factory) -> Path:
    from myosuite import make_env

    env = make_env(TASK)
    model = env.unwrapped.model
    names = [model.actuator(i).name for i in range(model.nu)]
    env.close()
    path = tmp_path_factory.mktemp("dynsyn") / "elbow.json"
    DynSynGroups(GROUPS, names).save(path)
    return path


def _runner(cfg) -> MjlabOnPolicyRunner:
    """An mjlab runner as the trainer builds it, without a log directory."""
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cpu"))
    runner = MjlabOnPolicyRunner(env, asdict(load_rl_cfg(TASK)), None, "cpu")
    track_action_map_in_checkpoints(runner)
    return runner


@pytest.fixture
def lone_checkpoint(groups_file, tmp_path) -> Path:
    """A DynSyn checkpoint without its run directory (as a W&B download), 2 envs, 100 steps."""
    cfg = _cfg(2)
    apply_action_map(cfg, DynSyn(DynSynGroups.load(groups_file), 0, 1000))
    runner = _runner(cfg)
    runner.env.unwrapped.common_step_counter = 100
    checkpoint = tmp_path / "wandb_checkpoints" / "model_1.pt"
    checkpoint.parent.mkdir()
    runner.save(str(checkpoint))
    runner.env.close()
    return checkpoint


def test_checkpoint_alone_records_groups_schedule_and_bound(lone_checkpoint) -> None:
    assert not (lone_checkpoint.parent / "params").exists()
    trained = trained_action_map(lone_checkpoint)
    m = trained.action_map
    assert m.groups.groups == GROUPS
    assert (m.start_steps, m.ramp_steps) == (0, 1000)
    assert (trained.env_steps, trained.saved_counter) == (200, 100)
    assert m.amplitude(trained.env_steps) == pytest.approx(0.02)
    with pytest.raises(ValueError, match="'dynsyn' action map"):
        load_rslrl_policy(lone_checkpoint, 6)


def test_resume_of_a_lone_checkpoint_with_other_num_envs_keeps_the_schedule(
    lone_checkpoint,
) -> None:
    cfg = SimpleNamespace(env=_cfg(1), groups=None)
    resolve_dynsyn(cfg, lone_checkpoint)
    runner = _runner(cfg.env)
    runner.load(str(lone_checkpoint))
    env = runner.env.unwrapped
    term = action_map_term(env)
    assert env.common_step_counter == 100
    assert term.env_steps == 200
    env.step(torch.full((1, 6), 0.3))
    assert term.env_steps == 201
    runner.env.close()


def test_plain_env_resume_of_a_dynsyn_checkpoint_fails(lone_checkpoint) -> None:
    runner = _runner(_cfg(2))  # same action width, so the weights would fit
    with pytest.raises(ValueError, match="'dynsyn' action map"):
        runner.load(str(lone_checkpoint))
    runner.env.close()


def test_resume_refuses_contradicting_groups(
    groups_file, lone_checkpoint, tmp_path
) -> None:
    other = DynSynGroups.load(groups_file)
    other.groups = other.groups[::-1]
    other_path = tmp_path / "other.json"
    other.save(other_path)
    cfg = SimpleNamespace(env=_cfg(2), groups=str(other_path))
    with pytest.raises(ValueError, match="differs from the groups"):
        resolve_dynsyn(cfg, lone_checkpoint)

    plain = _runner(_cfg(2))
    path = tmp_path / "plain" / "model_1.pt"
    path.parent.mkdir()
    plain.save(str(path))
    plain.env.close()
    with pytest.raises(ValueError, match="trained without DynSyn"):
        resolve_dynsyn(cfg, path)
    with pytest.raises(ValueError, match="Give --groups"):
        resolve_dynsyn(SimpleNamespace(env=_cfg(2), groups=None), None)


def test_short_run_saves_checkpoints_that_carry_the_map(groups_file, tmp_path) -> None:
    cfg = DynSynTrainConfig.from_task(TASK)
    cfg.env.scene.num_envs = 4
    cfg.agent.max_iterations = 2
    cfg.agent.save_interval = 1
    cfg.agent.logger = "tensorboard"
    cfg = DynSynTrainConfig(
        env=cfg.env,
        agent=cfg.agent,
        groups=str(groups_file),
        max_amplitude=0.0,
        device="cpu",
        log_root=str(tmp_path),
    )
    run = train(TASK, cfg)
    checkpoints = sorted(run.glob("model_*.pt"))
    assert checkpoints
    trained = trained_action_map(checkpoints[-1])
    assert trained.action_map.groups.groups == GROUPS
    assert trained.action_map.action_dim == 2
