# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""mjlab twin of the sensorimotor delay / observation noise (``SensorimotorCfg``).

A CPU registration carrying ``sensorimotor=`` gets an mjlab twin through
``register_cpu_twins`` (the muscle-condition path). On the elbow twin (4 envs,
9-step episodes so auto-resets happen, plus an explicit partial reset):

* the delayed actor observation is the undelayed one shifted by ``k`` per
  env and episode (against the same env's critic and an undelayed twin);
* an action delay equals the undelayed twin fed the shifted action stream;
* the noise is N(0, sigma^2), independent across envs, seeded by the torch RNG;
* both backends relate their delayed streams to their undelayed ones by the
  same index shift.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.tier2

import mjlab.tasks.registry as mjlab_registry  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers.observation_manager import ObservationTermCfg  # noqa: E402

import myosuite  # noqa: E402, F401
from myosuite.core import registry  # noqa: E402
from myosuite.core.sensorimotor import SensorimotorCfg  # noqa: E402
from myosuite.envs.myo.backends.mjlab.tasks import cpu_reference as ref  # noqa: E402
from myosuite.envs.myo.backends.mjlab.tasks.mdp import (  # noqa: E402
    DelayedObservationCfg,
)
from myosuite.envs.myo.backends.mjlab.tasks.pose.config.elbow import (  # noqa: E402
    elbow_pose_env_cfg,
    elbow_pose_ppo_runner_cfg,
)
from myosuite.envs.myo.backends.mjlab.tasks.registration import (  # noqa: E402
    register_cpu_twins,
)
from myosuite.terms.base_action import sigmoid_muscle_activation  # noqa: E402

BASE_ID = "myoElbowPose1D6MRandom-v0"
N_ENVS, EPISODE_STEPS, N_STEPS = 4, 9, 24
PARTIAL_STEP, PARTIAL_IDS = 5, (1,)
K_OBS, K_ACT, SIGMA = 3, 2, 0.05
CFGS = {
    "Base": None,
    "ObsDelay": SensorimotorCfg(obs_delay_steps=K_OBS),
    "ActDelay": SensorimotorCfg(action_delay_steps=K_ACT),
    "Noisy": SensorimotorCfg(obs_delay_steps=K_OBS, obs_noise_std=SIGMA),
    "Both": SensorimotorCfg(obs_delay_steps=K_OBS, action_delay_steps=K_ACT),
}


def _variant_id(name: str) -> str:
    return f"myoSmTest{name}ElbowPose1D6MRandom-v0"


@pytest.fixture(scope="module")
def twins() -> Iterator[dict[str, ManagerBasedRlEnv]]:
    """CPU registrations + mjlab twins of every :data:`CFGS` entry (removed after)."""
    spec = gym.spec(BASE_ID)
    ids = [_variant_id(name) for name in CFGS]
    for name, cfg in CFGS.items():
        kwargs = dict(spec.kwargs)
        if cfg is not None:
            kwargs["sensorimotor"] = cfg
        registry.register_env(
            _variant_id(name),
            entry_point=str(spec.entry_point),
            max_episode_steps=EPISODE_STEPS,
            kwargs=kwargs,
        )
    register_cpu_twins(tuple(ids), elbow_pose_env_cfg, elbow_pose_ppo_runner_cfg)
    envs = {}
    for name in CFGS:
        cfg = mjlab_registry.load_env_cfg(_variant_id(name))
        cfg.scene.num_envs = N_ENVS
        envs[name] = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    yield envs
    for env_id in ids:
        del gym.registry[env_id]
        registry._ENV_REGISTRY.pop(env_id, None)
        mjlab_registry._REGISTRY.pop(env_id, None)


def _actions(n: int, n_envs: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand((n, n_envs, 6), generator=gen) * 2 - 1


def _rollout(
    env: ManagerBasedRlEnv, actions: torch.Tensor, action_delay: int = 0
) -> dict[str, list[torch.Tensor]]:
    """Actor / critic obs, applied ctrl and reset flags per record.

    Record 0 is the reset; record ``t`` follows step ``t``. After step
    :data:`PARTIAL_STEP` envs :data:`PARTIAL_IDS` are reset explicitly. With
    ``action_delay`` the env is fed the action stream shifted by that many
    steps, zero after every reset of an env (the oracle of an action delay).
    """
    obs, _ = env.reset(seed=0)
    out = {"actor": [obs["actor"].clone()], "critic": [obs["critic"].clone()]}
    out["reset"] = [torch.ones(env.num_envs, dtype=torch.bool)]
    out["ctrl"] = [torch.zeros_like(actions[0])]
    queue = deque(torch.zeros_like(actions[0]) for _ in range(action_delay))
    for t, action in enumerate(actions, start=1):
        queue.append(action.clone())
        obs, _, term, trunc, _ = env.step(queue.popleft())
        out["ctrl"].append(
            env.action_manager.get_term("muscles").processed_action.clone()
        )
        reset = term | trunc
        if t == PARTIAL_STEP:
            obs, _ = env.reset(env_ids=torch.tensor(PARTIAL_IDS))
            reset[list(PARTIAL_IDS)] = True
        for pending in queue:
            pending[reset] = 0.0
        out["actor"].append(obs["actor"].clone())
        out["critic"].append(obs["critic"].clone())
        out["reset"].append(reset.clone())
    return out


def _shifted(
    seq: list[torch.Tensor], reset: list[torch.Tensor], k: int
) -> list[torch.Tensor]:
    """``seq`` delayed by ``k`` records per env, held at each env's last reset."""
    start = torch.zeros(seq[0].shape[0], dtype=torch.long)
    rows = torch.arange(seq[0].shape[0])
    stacked = torch.stack(seq)
    out = []
    for t in range(len(seq)):
        start = torch.where(reset[t], t, start)
        out.append(stacked[torch.maximum(start, torch.full_like(start, t - k)), rows])
    return out


def _assert_seq_equal(a: list[torch.Tensor], b: list[torch.Tensor]) -> None:
    np.testing.assert_array_equal(torch.stack(a).numpy(), torch.stack(b).numpy())


def test_twin_cfg_follows_cpu_registration(twins: dict[str, ManagerBasedRlEnv]) -> None:
    del twins
    full = SensorimotorCfg(
        obs_delay_steps=K_OBS, action_delay_steps=K_ACT, obs_noise_std=SIGMA
    )
    spec = gym.spec(BASE_ID)
    registry.register_env(
        _variant_id("Full"),
        entry_point=str(spec.entry_point),
        max_episode_steps=EPISODE_STEPS,
        kwargs={**spec.kwargs, "sensorimotor": full},
    )
    try:
        assert ref.cpu_task_spec(_variant_id("Full")).sensorimotor == full
        cfg = elbow_pose_env_cfg(_variant_id("Full"))
        actor, critic = cfg.observations["actor"], cfg.observations["critic"]
        assert actor.enable_corruption and not critic.enable_corruption
        for term in actor.terms.values():
            assert isinstance(term, DelayedObservationCfg)
            assert term.delay_steps == K_OBS and term.noise.std == SIGMA
        assert all(
            type(t) is ObservationTermCfg and t.noise is None
            for t in critic.terms.values()
        )
        assert cfg.actions["muscles"].action_delay_steps == K_ACT
    finally:
        del gym.registry[_variant_id("Full")]
        registry._ENV_REGISTRY.pop(_variant_id("Full"), None)
    base = elbow_pose_env_cfg(BASE_ID)
    assert not base.observations["actor"].enable_corruption
    assert all(
        type(t) is ObservationTermCfg for t in base.observations["actor"].terms.values()
    )
    assert base.actions["muscles"].action_delay_steps == 0


def test_obs_delay_shifts_actor_observation(
    twins: dict[str, ManagerBasedRlEnv],
) -> None:
    """Actor_t == critic_{t-k} (same env) == undelayed twin's actor_{t-k}, through resets."""
    actions = _actions(N_STEPS, N_ENVS)
    base = _rollout(twins["Base"], actions)
    delayed = _rollout(twins["ObsDelay"], actions)
    assert any(bool(r[0]) for r in delayed["reset"][1:]), "no auto-reset happened"
    _assert_seq_equal(delayed["reset"], base["reset"])
    _assert_seq_equal(delayed["critic"], base["actor"])  # dynamics untouched
    _assert_seq_equal(
        delayed["actor"], _shifted(delayed["critic"], delayed["reset"], K_OBS)
    )
    _assert_seq_equal(delayed["actor"], _shifted(base["actor"], base["reset"], K_OBS))


def test_action_delay_applies_shifted_actions(
    twins: dict[str, ManagerBasedRlEnv],
) -> None:
    """Action delay k == undelayed twin fed the shifted stream; ctrl_t = map(a_{t-k})."""
    actions = _actions(N_STEPS, N_ENVS)
    oracle = _rollout(twins["Base"], actions, action_delay=K_ACT)
    delayed = _rollout(twins["ActDelay"], actions)
    for key in ("reset", "actor", "critic", "ctrl"):
        _assert_seq_equal(delayed[key], oracle[key])
    start = 0  # record of env 0's last reset; its next k steps apply raw action 0
    for t in range(1, N_STEPS + 1):
        if delayed["reset"][t][0]:  # auto-reset zeroed processed_action inside step()
            start = t
            continue
        raw = actions[t - 1 - K_ACT, 0] if t - start > K_ACT else torch.zeros(6)
        expected = sigmoid_muscle_activation(raw.clamp(-1, 1), torch)
        np.testing.assert_allclose(
            delayed["ctrl"][t][0].numpy(), expected.numpy(), rtol=1e-6
        )


def test_obs_noise_statistics_and_seeding(twins: dict[str, ManagerBasedRlEnv]) -> None:
    """Noise after the delay: N(0, sigma^2), independent across envs, seeded."""
    actions = _actions(N_STEPS, N_ENVS)
    run = _rollout(twins["Noisy"], actions)
    rerun = _rollout(twins["Noisy"], actions)
    _assert_seq_equal(run["actor"], rerun["actor"])
    noise = torch.stack(run["actor"]) - torch.stack(
        _shifted(run["critic"], run["reset"], K_OBS)
    )
    n = noise.numel()
    assert abs(float(noise.std()) - SIGMA) < 0.1 * SIGMA
    assert abs(float(noise.mean())) < 4 * SIGMA / n**0.5
    per_env = noise.permute(1, 0, 2).reshape(N_ENVS, -1)
    corr = torch.corrcoef(per_env) - torch.eye(N_ENVS)
    assert float(corr.abs().max()) < 4 / per_env.shape[1] ** 0.5


def test_backends_share_index_shift_semantics(
    twins: dict[str, ManagerBasedRlEnv],
) -> None:
    """CPU and mjlab: same action stream, same delays -> the same shift relations."""
    actions = _actions(EPISODE_STEPS - 1, 1, seed=1)
    cfg = SensorimotorCfg(obs_delay_steps=K_OBS, action_delay_steps=K_ACT)

    def cpu_run(env: gym.Env, delay: int = 0) -> dict[str, list[torch.Tensor]]:
        out = {"obs": [torch.as_tensor(env.reset(seed=0)[0])[None]], "ctrl": []}
        queue = deque(np.zeros(6, np.float32) for _ in range(delay))
        for a in actions[:, 0].numpy():
            queue.append(a)
            out["obs"].append(torch.as_tensor(env.step(queue.popleft())[0])[None])
            out["ctrl"].append(torch.as_tensor(env.unwrapped.data.ctrl.copy())[None])
        return out

    # Oracle on each backend: the undelayed env fed the shifted action stream.
    cpu_delayed = cpu_run(gym.make(BASE_ID, sensorimotor=cfg))
    cpu_oracle = cpu_run(gym.make(BASE_ID), delay=K_ACT)
    cpu_reset = [torch.tensor([t == 0]) for t in range(len(cpu_oracle["obs"]))]
    _assert_seq_equal(cpu_delayed["obs"], _shifted(cpu_oracle["obs"], cpu_reset, K_OBS))
    _assert_seq_equal(cpu_delayed["ctrl"], cpu_oracle["ctrl"])

    batch = actions.expand(-1, N_ENVS, -1).clone()
    mj_delayed = _rollout(twins["Both"], batch)
    mj_oracle = _rollout(twins["Base"], batch, action_delay=K_ACT)
    _assert_seq_equal(
        mj_delayed["actor"], _shifted(mj_oracle["actor"], mj_oracle["reset"], K_OBS)
    )
    _assert_seq_equal(mj_delayed["ctrl"], mj_oracle["ctrl"])
    _assert_seq_equal(mj_delayed["critic"], mj_oracle["critic"])
    # The fill (raw 0) is the same excitation on both backends.
    for t in range(K_ACT):
        np.testing.assert_allclose(
            mj_delayed["ctrl"][t + 1][0].numpy(),
            cpu_delayed["ctrl"][t][0].numpy(),
            rtol=1e-6,
        )
