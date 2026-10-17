# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache 2 license in the root LICENSE file.
"""TERRA adapter: observation layout, terrain heights, reference composer, inference."""

import importlib.util
import warnings

import mujoco
import numpy as np
import pytest

from myosuite.core.model_builder import build_from_recipe
from myosuite.core.trajectory_io import MotionClip
from myosuite.integrations.musclemimic import (
    MIMIC_SITES,
    TerrainObsCfg,
    TerrainObservation,
    TerrainHeights,
    compose_waypoint_reference,
)
from myosuite.core.trajectory_io import motion_clip_from_states
from myosuite.integrations.musclemimic.fullbody_local_policy import (
    FullbodyStateAdapter,
    LocalPolicyArtifacts,
    LocalPolicyRunner,
)

pytestmark = pytest.mark.tier1

# The TERRA-4B observation settings (experiment.env_params of its saved config).
CONFIG = {
    "experiment": {
        "env_params": {
            "use_egocentric_root_observations": True,
            "enable_global_root_position_observation": False,
            "preserve_trajectory_root_xy": True,
            "enable_heightmap_observations": True,
            "heightmap_grid_rows": 11,
            "heightmap_grid_cols": 11,
            "heightmap_grid_resolution": 0.1,
            "heightmap_grid_forward_offset": 0.0,
            "heightmap_body_name": "pelvis",
            "enable_touch_sensor_observations": True,
            "enable_joint_pos_observations": True,
            "enable_joint_vel_observations": True,
            "enable_muscle_length_observations": False,
            "enable_muscle_velocity_observations": False,
            "enable_muscle_force_observations": False,
            "enable_muscle_excitation_observations": True,
            "enable_muscle_activation_observations": True,
            "goal_type": "TerraFullBodyTrackingGoal",
            "goal_params": {
                "lookahead_steps": [1, 20, 40, 60, 80],
                "sites_for_mimic": list(MIMIC_SITES),
            },
        }
    }
}


def _scene(spec: mujoco.MjSpec) -> None:
    spec.worldbody.add_geom(
        name="step",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        group=2,
        pos=[0, -1.5, 0.05],
        size=[0.5, 0.2, 0.05],
    )


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    # Layout only: the native MyoFullBody has the TERRA actor's joints, sites and sensors.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_from_recipe("musclemimic_fullbody", _scene)[0]


def _walk_clip(model: mujoco.MjModel, frames: int = 400) -> MotionClip:
    q = np.repeat(model.qpos0[None], frames, 0)
    t = np.arange(frames) / 100
    q[:, 1] -= t  # 1 m/s toward -y
    q[:, 7:] += 0.1 * np.sin(2 * np.pi * t)[:, None]  # 1 s gait cycle
    return MotionClip(
        qpos=q, qvel=None, site_xpos=None, site_names=None, frequency_hz=100.0
    )


def _reference(model: mujoco.MjModel, data: mujoco.MjData) -> MotionClip:
    heights = TerrainHeights(model)
    return compose_waypoint_reference(
        model,
        _walk_clip(model),
        np.zeros(2),
        np.array([[0, -2.5], [1, -2.5]]),
        0.01,
        lambda xy: heights(data, xy),
        MIMIC_SITES,
    )


def test_config_is_read_and_checked() -> None:
    cfg = TerrainObsCfg.from_config(CONFIG)
    assert cfg == TerrainObsCfg()
    bad = {
        "experiment": {
            **CONFIG["experiment"],
            "env_params": {
                **CONFIG["experiment"]["env_params"],
                "goal_type": "TerraGoal",
            },
        }
    }
    with pytest.raises(ValueError):
        TerrainObsCfg.from_config(bad)


def test_terrain_heights_hit_only_terrain(model: mujoco.MjModel) -> None:
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[:2] = [0, -1.5]  # the actor stands on the step: rays must ignore it
    mujoco.mj_forward(model, data)
    h = TerrainHeights(model)(data, [[0, -1.5], [0.4, -1.4], [0.6, -1.5], [3, 3]])
    np.testing.assert_allclose(h, [0.1, 0.1, 0.0, 0.0], atol=1e-9)


def test_observation_layout(model: mujoco.MjModel) -> None:
    data = mujoco.MjData(model)
    ref = _reference(model, data)
    data.qpos[:], data.qvel[:] = ref.qpos[0], ref.qvel[0]
    data.ctrl[:] = 0.3
    mujoco.mj_forward(model, data)
    observe = TerrainObservation(model)
    obs = observe(data, ref, 0)
    nq, nv, nu = model.nq, model.nv, model.nu
    goal = 452
    assert (
        obs.shape == ((nq - 3) + nv + 2 * nu + 4 + 121 + goal,)
        and obs.dtype == np.float32
    )
    assert obs[0] == np.float32(data.qpos[2])
    np.testing.assert_allclose(obs[1:4], [0, 0, -1], atol=1e-6)
    muscle = obs[nq - 3 + nv : nq - 3 + nv + 2 * nu]
    np.testing.assert_allclose(
        muscle[0::2], 0.3, rtol=1e-6
    )  # excitation, then activation
    # Heights relative to the pelvis: flat floor at the start, about -pelvis height.
    hm = obs[-goal - 121 : -goal]
    assert np.allclose(hm, -data.xpos[model.body("pelvis").id, 2], atol=1e-6)


def test_reference_follows_route_and_terrain(model: mujoco.MjModel) -> None:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ref = _reference(model, data)
    root = ref.qpos[:, :2]
    assert np.linalg.norm(root[0]) < 0.1
    assert np.linalg.norm(root[-1] - [1.5, -2.5]) < 0.2  # 0.5 m past the last waypoint
    near = [np.min(np.linalg.norm(root - p, axis=1)) for p in ([0, -2.5], [1, -2.5])]
    assert max(near) < 0.35  # corners are rounded over 0.3 m
    speed = np.linalg.norm(ref.qvel[:, :2], axis=1)
    assert abs(np.median(speed) - 1.0) < 0.05
    on_step = np.abs(root[:, 1] + 1.5) < 0.05
    lift = ref.qpos[on_step, 2] - model.qpos0[2]
    assert np.all(lift > 0.05)  # root raised over the 10 cm step
    np.testing.assert_allclose(np.linalg.norm(ref.qpos[:, 3:7], axis=1), 1, atol=1e-9)


def _random_policy(
    obs_dim: int, nu: int, rng: np.random.Generator
) -> LocalPolicyRunner:
    def dense(n_in: int, n_out: int) -> dict:
        return {
            "kernel": rng.normal(0, 1 / np.sqrt(n_in), (n_in, n_out)).astype(
                np.float32
            ),
            "bias": np.zeros(n_out, np.float32),
        }

    ln = lambda n: {"scale": np.ones(n, np.float32), "bias": np.zeros(n, np.float32)}  # noqa: E731
    actor = {
        "block0_layer0_dense": dense(obs_dim, 64),
        "block0_layer0_ln": ln(64),
        "block0_layer1_dense": dense(64, 32),
        "block0_layer1_ln": ln(32),
        "block0_proj": dense(obs_dim, 32),
        "res_gate_0": np.array([-2.0], np.float32),
        "tail_dense": dense(32, 32),
        "tail_ln": ln(32),
        "output": dense(32, nu),
    }
    params = {"actor": actor, "log_std": np.full(nu, np.log(0.1), np.float32)}
    return LocalPolicyRunner(
        LocalPolicyArtifacts(
            params,
            rng.normal(size=obs_dim).astype(np.float32),
            np.ones(obs_dim, np.float32),
            np.array(1e9, np.float32),
            obs_dim,
            nu,
        ),
        stochastic=True,
        seed=0,
        update_normalizer=False,
    )


def test_policy_inference_drives_the_env(model: mujoco.MjModel) -> None:
    from myosuite.envs.waypoint import WaypointEnv, WaypointTaskCfg

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ref = _reference(model, data)
    observe = TerrainObservation(model)
    rng = np.random.default_rng(0)
    policy = _random_policy(observe(data, ref, 0).size, model.nu, rng)
    env = WaypointEnv(
        model,
        task=WaypointTaskCfg(waypoints=((0, -2.5), (1, -2.5))),
        initial_qpos=ref.qpos[0],
        initial_qvel=ref.qvel[0],
    )
    env.reset(seed=0)
    np.testing.assert_array_equal(env.data.qvel, ref.qvel[0])
    for step in range(3):
        policy.stochastic = False
        mean = policy.action_from_obs(observe(env.data, ref, step))
        policy.stochastic = True
        action = policy.action_from_obs(observe(env.data, ref, step))
        assert action.shape == (model.nu,) and np.all(np.abs(action) <= 1)
        assert not np.array_equal(mean, action)
        env.step(action)
    with pytest.raises(ValueError):
        policy.action_from_obs(np.zeros(3))


@pytest.mark.skipif(
    importlib.util.find_spec("musclemimic_models") is None,
    reason="needs musclemimic_models",
)
def test_terra_actor() -> None:
    from myosuite.integrations.musclemimic import build_terrain_fullbody_spec

    spec = build_terrain_fullbody_spec(_scene)
    model = spec.compile()
    assert model.opt.iterations == 4 and model.opt.ls_iterations == 8
    assert model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    assert model.geom("GasLat_at_shank_r_wrap").size[0] == pytest.approx(0.052)
    assert model.site("BIClong_ellipsoid_BIClong_2_sidesite").pos[1] == pytest.approx(
        0.020
    )
    # musclemimic_models 1.0.5 knee coupling (fixed with the opposite sign in 1.0.6).
    eq = model.eq("knee_angle_translation2_constraint_l")
    assert eq.data[1] < 0
    TerrainObservation(model)


@pytest.mark.skipif(
    importlib.util.find_spec("musclemimic_models") is None,
    reason="needs musclemimic_models",
)
def test_matches_upstream_terra_rollout() -> None:
    """Same states and observations as TERRA's own env (``scripts/terra_parity``).

    The older compact-goal fixture checks physics and the shared physical state
    prefix; its obsolete goal is excluded. Tracking goals use a separate fixture.
    """
    from pathlib import Path

    from myosuite.envs.waypoint import WaypointEnv, WaypointTaskCfg
    from myosuite.integrations.musclemimic import (
        TerrainController,
        build_terrain_fullbody_spec,
    )

    up = np.load(Path(__file__).parent / "data" / "terra_upstream_obs.npz")
    model = build_terrain_fullbody_spec(disabled_contact_pairs=()).compile()
    ref = motion_clip_from_states(model, up["ref_qpos"], 0.01, MIMIC_SITES)

    class Replay:
        obs_adapter = TerrainObservation(model, reference=ref)
        seen: list = []
        state_adapter = FullbodyStateAdapter(
            model,
            {
                "enable_muscle_length_observations": False,
                "enable_muscle_velocity_observations": False,
                "enable_muscle_force_observations": False,
            },
        )

        def action_for(self, data, clip, frame_idx):
            self.seen.append(
                np.concatenate(
                    [
                        self.state_adapter.build_state(data),
                        self.obs_adapter.heightmap(data),
                    ]
                )
            )
            return up["actions"][len(self.seen) - 1]

    policy = Replay()
    env = WaypointEnv(
        model,
        task=WaypointTaskCfg(waypoints=((1.0, -2.5),)),
        initial_qpos=ref.qpos[0],
        initial_qvel=ref.qvel[0],
    )
    env.reset(seed=0)
    controller = TerrainController(policy, model, ref, env.frame_skip)
    for qpos in up["qpos"]:
        np.testing.assert_allclose(env.data.qpos, qpos, atol=1e-6)
        env.step(controller(env.data))
    nq, nv, nu = model.nq, model.nv, model.nu
    touch = slice((nq - 2) + nv + 2 * nu, (nq - 2) + nv + 2 * nu + 4)
    seen = np.array(policy.seen)
    np.testing.assert_allclose(
        seen[:, touch], up["obs"][:, touch], atol=1e-2
    )  # newtons
    rest = np.ones(seen.shape[1], bool)
    rest[touch] = False
    np.testing.assert_allclose(
        seen[:, rest], up["obs"][:, : seen.shape[1]][:, rest], atol=1e-4
    )


def test_tracking_checkpoint_layout(model):
    from copy import deepcopy

    config = deepcopy(CONFIG)
    params = config["experiment"]["env_params"]
    params["goal_type"] = "TerraFullBodyTrackingGoal"
    params["use_egocentric_root_observations"] = True
    params["goal_params"]["lookahead_steps"] = [1, 20, 40, 60, 80]
    cfg = TerrainObsCfg.from_config(config)
    ref = motion_clip_from_states(
        model, np.repeat(model.qpos0[None], 100, axis=0), 0.01, MIMIC_SITES
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    obs = TerrainObservation(model, cfg)(data, ref, 0)
    # 204 immediate errors plus four 62-component future targets.
    goal = obs[-452:]
    assert obs.size == (model.nq - 3) + model.nv + 2 * model.nu + 4 + 121 + 452
    np.testing.assert_allclose(goal[:204], 0, atol=1e-6)
    np.testing.assert_allclose(goal[264:266], [0.2, 1])
    np.testing.assert_allclose(TerrainObservation(model, cfg)(data, ref, 99)[-1], 0)


@pytest.mark.parametrize(
    "dt,route,stationary",
    [
        (0, [[0, -1]], False),
        (0.01, [], False),
        (0.01, [[0, -1], [0, -1]], False),
        (0.01, [[0, -1]], True),
    ],
)
def test_invalid_reference_inputs_are_rejected(model, dt, route, stationary):
    clip = _walk_clip(model)
    if stationary:
        clip.qpos[:, :2] = 0
    with pytest.raises(ValueError):
        compose_waypoint_reference(
            model,
            clip,
            np.zeros(2),
            np.asarray(route),
            dt,
            lambda xy: np.zeros(len(xy)),
            MIMIC_SITES,
        )


@pytest.mark.skipif(
    importlib.util.find_spec("musclemimic_models") is None,
    reason="needs musclemimic_models",
)
def test_tracking_goals_match_pinned_upstream_assembly() -> None:
    from pathlib import Path
    from myosuite.integrations.musclemimic import build_terrain_fullbody_spec

    fixture = np.load(Path(__file__).parent / "data" / "terra_tracking_goals.npz")
    model = build_terrain_fullbody_spec(disabled_contact_pairs=()).compile()
    ref = motion_clip_from_states(model, fixture["ref_qpos"], 0.01, MIMIC_SITES)
    observe = TerrainObservation(model)
    data = mujoco.MjData(model)
    for qpos, qvel, frame, expected in zip(
        fixture["qpos"], fixture["qvel"], fixture["frames"], fixture["goals"]
    ):
        data.qpos[:], data.qvel[:] = qpos, qvel
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(
            observe.goal(data, ref, int(frame)), expected, atol=1e-6, rtol=1e-6
        )
