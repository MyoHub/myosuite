# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The mjlab MuscleMimic policy bridge builds its observation on the sim device.

A real full-body mjlab env runs on CPU warp with a synthetic gait clip (sites
from MuJoCo forward kinematics).  The batched (``obs_backend="torch"``) and the
reference per-env CPU ``mj_forward`` (``obs_backend="cpu"``) observations of the
same post-step states agree to float32 tolerance, through resets and clip ends,
and the batched path neither runs a CPU forward nor syncs the host per env.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from myosuite.tests.support.optional_deps import (
    require_mjlab,
    require_mujoco_warp,
    require_musclemimic_models,
)

pytestmark = pytest.mark.tier2

torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

_ENTITY = "mimic_fullbody_robot"
_N_ENVS = 4
_N_FRAMES = 40
_GOAL = {"n_step_lookahead": 5, "n_step_stride": 1, "use_concise_lookahead": True}
# MjData fields FullbodyObsAdapter.build reads.
_SIM_FIELDS = (
    "qpos",
    "qvel",
    "ctrl",
    "act",
    "actuator_length",
    "actuator_velocity",
    "actuator_force",
    "sensordata",
    "site_xpos",
    "site_xmat",
    "cvel",
    "subtree_com",
)


@pytest.fixture(scope="module", autouse=True)
def _require_stack() -> None:
    require_mjlab()
    require_mujoco_warp()
    require_musclemimic_models()


def _write_gait_clip(model: Any, path: Path) -> Path:
    """Walking-like clip NPZ in the MuscleMimic trajectory layout."""
    rng = np.random.default_rng(0)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    base, dt = data.qpos.copy(), 0.01
    t = np.arange(_N_FRAMES) * dt
    qpos = np.repeat(base[None], _N_FRAMES, axis=0)
    qpos[:, 0] += 1.1 * t
    for i in range(_N_FRAMES):
        mujoco.mju_euler2Quat(
            qpos[i, 3:7], np.array([0.03, 0.05 * np.sin(11.3 * t[i]), 0.2]), "xyz"
        )
    for jid in range(1, model.njnt):
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_HINGE):
            adr = int(model.jnt_qposadr[jid])
            wave = rng.uniform(0.02, 0.3) * np.sin(5.7 * t + rng.uniform(0, 6.3))
            lo, hi = model.jnt_range[jid] if model.jnt_limited[jid] else (-9, 9)
            qpos[:, adr] = np.clip(base[adr] + wave, lo, hi)
    qvel = np.zeros((_N_FRAMES, model.nv))
    for i in range(1, _N_FRAMES):
        mujoco.mj_differentiatePos(model, qvel[i], dt, qpos[i - 1], qpos[i])
    qvel[0] = qvel[1]
    rec: dict[str, list[np.ndarray]] = {
        k: [] for k in ("site_xpos", "site_xmat", "cvel", "subtree_com")
    }
    for i in range(_N_FRAMES):
        data.qpos[:], data.qvel[:] = qpos[i], qvel[i]
        mujoco.mj_forward(model, data)
        for key, values in rec.items():
            values.append(getattr(data, key).copy())
    np.savez(
        path,
        qpos=qpos,
        qvel=qvel,
        **{key: np.asarray(values) for key, values in rec.items()},
        site_bodyid=np.asarray(model.site_bodyid),
        body_rootid=np.asarray(model.body_rootid),
        site_names=np.asarray([model.site(i).name for i in range(model.nsite)]),
        frequency=np.asarray(1.0 / dt),
    )
    return path


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    from ml_collections import config_dict
    from mjlab.envs import ManagerBasedRlEnv

    from myosuite.core.trajectory_io import load_motion_clip
    from myosuite.envs.myo.backends.mjlab import mimic_mjlab_env as mimic
    from myosuite.integrations.musclemimic.fullbody_local_policy import (
        FullbodyObsAdapter,
    )
    from myosuite.integrations.musclemimic.fullbody_model import (
        FULLBODY_BODY2SITES_FOR_MIMIC,
        build_mimic_fullbody_spec,
        compile_mimic_fullbody_mjmodel,
        default_mimic_fullbody_config,
    )

    model, _, _ = compile_mimic_fullbody_mjmodel(default_mimic_fullbody_config())
    path = _write_gait_clip(model, tmp_path_factory.mktemp("clip") / "gait.npz")
    clip = load_motion_clip(path, expected_nq=model.nq, expected_nv=model.nv)
    adapter = FullbodyObsAdapter(
        model,
        clip,
        {"sites_for_mimic": list(FULLBODY_BODY2SITES_FOR_MIMIC.values()), **_GOAL},
    )
    cfg = config_dict.create(**dict(default_mimic_fullbody_config()))
    env_cfg = mimic._make_mimic_env_cfg(
        _task_id="test-mimic-fullbody-bridge",
        entity_name=_ENTITY,
        variant="fullbody",
        spec_fn=lambda: build_mimic_fullbody_spec(cfg)[0],
        muscle_actuators=mimic._muscle_actuator_names(model),
        tendon_targets=mimic._muscle_tendon_names(model),
        sim_dt=float(cfg.sim_dt),
        ctrl_dt=float(cfg.ctrl_dt),
        max_episode_steps=int(cfg.max_episode_steps),
        clip=clip,
        num_envs=_N_ENVS,
        mj_model=model,
        action_mode="direct",
    )
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    yield {"env": env, "model": model, "clip": clip, "adapter": adapter}
    env.close()


def _cpu_reference_obs(setup: dict[str, Any], frames: np.ndarray) -> np.ndarray:
    """``obs_backend="cpu"`` semantics: CPU ``mj_forward`` of each env's state."""
    model, adapter, sim = setup["model"], setup["adapter"], setup["env"].sim.data
    data = mujoco.MjData(model)
    out = []
    for e, frame in enumerate(frames):
        for field in ("qpos", "qvel", "ctrl", "act"):
            getattr(data, field)[:] = getattr(sim, field)[e].numpy()
        mujoco.mj_forward(model, data)
        out.append(adapter.build(data, int(frame)))
    return np.stack(out)


def _builder_on_sim_arrays(setup: dict[str, Any], frames: np.ndarray) -> np.ndarray:
    """CPU ``FullbodyObsAdapter`` fed mjlab's own post-step arrays (same inputs)."""
    from types import SimpleNamespace

    sim = setup["env"].sim.data
    return np.stack(
        [
            setup["adapter"].build(
                SimpleNamespace(**{f: getattr(sim, f)[e].numpy() for f in _SIM_FIELDS}),
                int(frame),
            )
            for e, frame in enumerate(frames)
        ]
    )


def _site_path_actuators(model: Any) -> np.ndarray:
    """Actuators whose tendon is a pure site path (no sphere/cylinder wrap).

    Wrapped paths are not comparable to float64 MuJoCo at 1e-4: mujoco_warp's
    float32 inside-wrap solve falls back to an approximate point on some states
    (mm errors), and side/tangency branch flips are discontinuous.
    """
    wrap = {int(mujoco.mjtWrap.mjWRAP_SPHERE), int(mujoco.mjtWrap.mjWRAP_CYLINDER)}
    keep = []
    for a in range(model.nu):
        t = int(model.actuator_trnid[a, 0])
        adr, num = model.tendon_adr[t], model.tendon_num[t]
        if not wrap.intersection(int(w) for w in model.wrap_type[adr : adr + num]):
            keep.append(a)
    return np.asarray(keep)


@pytest.fixture(scope="module")
def rollout(setup: dict[str, Any]) -> dict[str, Any]:
    """Default-backend bridge obs and both references over a seeded rollout."""
    from myosuite.integrations.musclemimic.mjlab_onnx_policy import (
        _FullbodyMjlabPolicyBridge,
    )

    env = setup["env"]
    bridge = _FullbodyMjlabPolicyBridge(
        env=env,
        cpu_model=setup["model"],
        obs_adapter=setup["adapter"],
        clip=setup["clip"],
    )
    torch.manual_seed(0)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    act_dim = sum(env.action_manager.action_term_dim)
    rec: dict[str, Any] = {"obs": [], "sim_ref": [], "cpu_ref": [], "resets": 0}
    for _ in range(_N_FRAMES):
        frames = np.asarray(bridge._current_frame_indices())
        rec["resets"] += int((env.episode_length_buf == 0).sum())
        rec["obs"].append(np.asarray(bridge._build_fullbody_obs_batch()))
        rec["sim_ref"].append(_builder_on_sim_arrays(setup, frames))
        rec["cpu_ref"].append(_cpu_reference_obs(setup, frames))
        action = rng.uniform(0.0, 0.5, (env.num_envs, act_dim))
        env.step(torch.as_tensor(action, dtype=torch.float32))
    return {k: np.concatenate(v) if isinstance(v, list) else v for k, v in rec.items()}


def test_obs_equals_the_cpu_builder_on_the_sim_arrays(
    rollout: dict[str, Any],
) -> None:
    """The bridge builds the reference observation from mjlab's own state.

    Covers fresh resets and clip ends (episodes restart within the rollout).
    """
    assert rollout["resets"] > _N_ENVS  # initial reset plus later auto-resets
    np.testing.assert_allclose(rollout["obs"], rollout["sim_ref"], rtol=1e-5, atol=1e-5)


def test_obs_matches_cpu_forward_of_the_same_state(
    setup: dict[str, Any], rollout: dict[str, Any]
) -> None:
    """Kinematic blocks agree with a CPU ``mj_forward`` of the same state.

    Muscle fields are compared on pure site-path tendons; touch values come from
    the two contact solvers and are only checked through the test above.
    """
    adapter, model = setup["adapter"], setup["model"]
    n_state = 5 + len(adapter._qpos_non_root_ind) + 6 + len(adapter._qvel_non_root_ind)
    n_touch = len(adapter._touch_sensor_ids)
    muscle0 = n_state
    goal0 = muscle0 + 5 * model.nu + n_touch
    obs, ref = rollout["obs"], rollout["cpu_ref"]
    kinematic = np.r_[0:n_state, goal0 : obs.shape[1]]
    np.testing.assert_allclose(obs[:, kinematic], ref[:, kinematic], atol=1e-4)
    muscles = (muscle0 + 5 * _site_path_actuators(model))[:, None] + np.arange(5)
    np.testing.assert_allclose(
        obs[:, muscles.ravel()], ref[:, muscles.ravel()], rtol=1e-4, atol=1e-4
    )


def test_policy_call_skips_cpu_forward_and_per_env_syncs(
    setup: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default policy path runs no CPU ``mj_forward`` and at most one host sync.

    The one sync is ``ClipTrajectorySource.update``'s reset check.
    """
    from myosuite.integrations.musclemimic.mjlab_onnx_policy import (
        FullbodyOrbaxMjlabPolicy,
    )
    from myosuite.tests.support.host_sync import HostSyncCounter

    calls = {"n": 0}
    real_forward = mujoco.mj_forward

    def _counting_forward(*args: Any) -> None:
        calls["n"] += 1
        real_forward(*args)

    monkeypatch.setattr(mujoco, "mj_forward", _counting_forward)
    env, model = setup["env"], setup["model"]
    obs_dim = _obs_dim(setup)
    common = dict(
        env=env,
        cpu_model=model,
        obs_adapter=setup["adapter"],
        clip=setup["clip"],
        artifacts=_tiny_artifacts(obs_dim, model.nu),
    )
    calls["n"] = 0
    policy = FullbodyOrbaxMjlabPolicy(**common)
    dummy = torch.zeros(env.num_envs, 1)
    policy(dummy)
    with HostSyncCounter() as syncs:
        action = policy(dummy)
    assert calls["n"] == 0
    assert syncs.total <= 1, dict(syncs.counts)
    assert tuple(action.shape) == (env.num_envs, model.nu)

    reference = FullbodyOrbaxMjlabPolicy(**common, obs_backend="cpu")
    calls["n"] = 0
    reference(dummy)
    assert calls["n"] == env.num_envs  # the counter sees the reference path


def test_onnx_policy_runs_all_envs_in_one_batch(
    setup: dict[str, Any], tmp_path: Path
) -> None:
    """``env_indices`` runs every env through one ONNX call with one host copy each way."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from myosuite.integrations.musclemimic.actor_onnx import export_to_onnx
    from myosuite.integrations.musclemimic.actor_torch import make_actor_module
    from myosuite.integrations.musclemimic.mjlab_onnx_policy import (
        FullbodyOnnxMjlabPolicy,
    )
    from myosuite.tests.support.host_sync import HostSyncCounter

    env, model = setup["env"], setup["model"]
    obs_dim = _obs_dim(setup)
    artifacts = _tiny_artifacts(obs_dim, model.nu)
    actor = make_actor_module(artifacts)
    onnx_path = export_to_onnx(actor, tmp_path / "actor.onnx")
    policy = FullbodyOnnxMjlabPolicy(
        env,
        model,
        setup["adapter"],
        onnx_path,
        setup["clip"],
        env_indices=range(env.num_envs),
        output_ctrl=True,
    )
    dummy = torch.zeros(env.num_envs, 1)
    policy(dummy)  # the first call also resolves the clip source
    with HostSyncCounter() as syncs:
        action = policy(dummy)
    # cpu + numpy of the obs batch, upload of the actions, the clip-source check.
    assert syncs.counts["cpu"] == 1 and syncs.counts["H2D as_tensor"] == 1
    assert syncs.total <= 4, dict(syncs.counts)
    expected = actor(policy._build_fullbody_obs_batch()).clamp(-1.0, 1.0)
    ctrl = torch.as_tensor(model.actuator_ctrlrange, dtype=torch.float32)
    expected = torch.clamp(expected, ctrl[:, 0], ctrl[:, 1])
    assert tuple(action.shape) == (env.num_envs, model.nu)
    torch.testing.assert_close(action, expected.detach(), rtol=1e-4, atol=1e-5)


def _obs_dim(setup: dict[str, Any]) -> int:
    model = setup["model"]
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return int(setup["adapter"].build(data, 0).shape[0])


def _tiny_artifacts(obs_dim: int, action_dim: int) -> Any:
    from myosuite.integrations.musclemimic.fullbody_local_policy import (
        LocalPolicyArtifacts,
    )

    rng = np.random.default_rng(1)
    dims = (obs_dim, 32, action_dim)
    actor = {
        f"Dense_{i}": {
            "kernel": rng.normal(0, dims[i] ** -0.5, dims[i : i + 2]).astype(
                np.float32
            ),
            "bias": np.zeros(dims[i + 1], np.float32),
        }
        for i in range(2)
    }
    return LocalPolicyArtifacts(
        params={"actor": actor},
        obs_mean=np.zeros(obs_dim, np.float32),
        obs_var=np.ones(obs_dim, np.float32),
        obs_count=np.asarray(1e-6, np.float32),
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
