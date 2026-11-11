"""Random finger-reach targets are sampled from the reachable workspace on both backends."""

from __future__ import annotations

import numpy as np
import pytest
import mujoco

from myosuite.utils.reach_workspace import reachable_target_points
from myosuite import make_env
from myosuite.utils import reach_workspace

pytestmark = pytest.mark.tier1

_IDS = ("myoFingerReachRandom-v0", "motorFingerReachRandom-v0")
_BOX = (np.array([[0.1, -0.1, 0.1]]), np.array([[0.27, 0.1, 0.3]]))


@pytest.mark.parametrize("env_id", _IDS)
def test_cpu_targets_come_from_the_reachable_table(env_id: str) -> None:
    env = make_env(env_id)
    u = env.unwrapped
    assert u.target_sampling == "workspace"
    points = u._workspace_points.reshape(-1, 3)
    low, high = _BOX
    assert ((points >= low) & (points <= high)).all()
    for seed in range(5):
        env.reset(seed=seed)
        target = u.data.site_xpos[u.target_sids[0]]
        assert np.abs(points - target).sum(axis=1).min() < 1e-9
    env.close()


def test_table_is_deterministic_and_covers_most_of_the_box() -> None:
    u = make_env("motorFingerReachRandom-v0").unwrapped
    sites = [u.model.site("IFtip").id]
    a = reachable_target_points(u.model, sites, *_BOX)
    b = reachable_target_points(u.model, sites, *_BOX, seed=0)
    np.testing.assert_array_equal(a, b)
    assert 20_000 < len(a)  # a large share of the joint space lands in the box
    assert (np.ptp(a, axis=0) > 0.1).all()  # the targets spread over the box


@pytest.mark.parametrize("env_id", _IDS)
def test_mjlab_twin_samples_the_same_table(env_id: str) -> None:
    pytest.importorskip("mjlab")
    from myosuite.tests.test_mjlab_cpu_twins import _make_pair

    cpu, mj = _make_pair(env_id)
    command = mj.command_manager.get_term("reach")
    assert type(command).__name__ == "WorkspaceReachTargetCommand"
    cpu.reset(seed=0)
    table = cpu.unwrapped._workspace_points.reshape(-1, 3)
    np.testing.assert_allclose(command._points.cpu().numpy(), table, atol=1e-5)
    mj.reset()
    target = command.command.cpu().numpy()[0]
    assert np.abs(table - target).sum(axis=1).min() < 1e-5


def test_mjlab_workspace_table_uses_scene_site_ids() -> None:
    """The table follows the robot's tip sites when another entity's site comes first.

    The table is built on the scene model, whose site ids are shifted by every site
    of an entity attached before the robot; entity-local tip ids pick other sites.
    """
    pytest.importorskip("mjlab")
    import mujoco
    from mjlab.entity import EntityCfg
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import myosuite.envs.myo.backends.mjlab  # noqa: F401 (registers twins)

    def _marker_spec() -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name="marker", pos=[0.5, 0.5, 0.5]).add_site(name="dot")
        return spec

    env_id = "motorFingerReachRandom-v0"
    cfg = load_env_cfg(env_id)
    cfg.scene.num_envs = 1
    cfg.scene.entities = {"marker": EntityCfg(spec_fn=_marker_spec)} | dict(
        cfg.scene.entities
    )
    mj = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    command = mj.command_manager.get_term("reach")
    robot = mj.scene[command.cfg.entity_name]
    local = [int(i) for i in command._tip_ids]
    assert [int(robot.indexing.site_ids[i]) for i in local] != local
    table = make_env(env_id).unwrapped._workspace_points.reshape(-1, 3)
    np.testing.assert_allclose(command._points.cpu().numpy(), table, atol=1e-5)
    mj.close()


def test_fixed_and_other_random_reach_envs_keep_box_sampling() -> None:
    for env_id in (
        "myoFingerReachFixed-v0",
        "myoHandReachRandom-v0",
        "myoArmReachRandom-v0",
    ):
        assert make_env(env_id).unwrapped.target_sampling == "box"


def _hinge_model(active: bool = False) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        f"""<mujoco><compiler angle="radian"/><worldbody><body>
        <joint name="hinge" range="-1 1" ref=".1"/>
        <geom size=".1"/><site name="tip" pos="1 0 0"/>
        </body></worldbody><equality>
        <joint joint1="hinge" active="{str(active).lower()}" polycoef=".2 0 0 0 0"/>
        </equality></mujoco>"""
    )


def _points(model: mujoco.MjModel) -> np.ndarray:
    return reachable_target_points(
        model,
        [model.site("tip").id],
        np.full((1, 3), -3),
        np.full((1, 3), 3),
        n_samples=128,
    )


def test_locked_joint_respects_its_nonzero_reference() -> None:
    points = _points(_hinge_model(active=True))[:, 0]
    np.testing.assert_allclose(points, np.tile([np.cos(0.2), np.sin(0.2), 0], (128, 1)))


def test_nonlinear_coupling_rejects_out_of_limit_followers() -> None:
    model = mujoco.MjModel.from_xml_string(
        """<mujoco><compiler angle="radian"/><worldbody><body>
        <joint name="master" ref=".2" range="-1 1"/>
        <geom size=".1"/><site name="master_tip" pos="1 0 0"/>
        <body pos="1 0 0"><joint name="follower" ref=".1" range=".1 .35"/>
        <geom size=".1"/><site name="tip" pos="1 0 0"/></body>
        </body></worldbody><equality>
        <joint joint1="follower" joint2="master" polycoef="0 0 2 0 0"/>
        </equality></mujoco>"""
    )
    points = reachable_target_points(
        model,
        [model.site("master_tip").id, model.site("tip").id],
        np.full((2, 3), -3),
        np.full((2, 3), 3),
        n_samples=128,
    )
    assert 0 < len(points) < 128
    master = np.arctan2(points[:, 0, 1], points[:, 0, 0])
    link = points[:, 1] - points[:, 0]
    follower = np.arctan2(link[:, 1], link[:, 0]) - master
    np.testing.assert_allclose(follower, 2 * master**2, atol=1e-14)
    assert np.all((follower >= 0) & (follower <= 0.25))


def test_locked_joint_outside_its_limits_has_no_workspace() -> None:
    model = _hinge_model(active=True)
    model.eq_data[0, 0] = 2
    with pytest.raises(ValueError, match="no joint configuration"):
        _points(model)


def test_inactive_equality_preserves_unconstrained_sampling() -> None:
    model = _hinge_model()
    points = _points(model)[:, 0]
    angles = np.random.default_rng(0).uniform(-1, 1, 128) - 0.1
    expected = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(128)))
    np.testing.assert_allclose(points, expected, atol=1e-15)


@pytest.mark.parametrize(
    "field", ["axis", "anchor", "site_body", "active", "coef", "limited"]
)
def test_cache_tracks_kinematics_and_in_place_model_edits(
    field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reach_workspace, "_CACHE", {})
    model = _hinge_model(active=field == "coef")
    cached = _points(model)
    assert _points(model) is cached
    before = cached.copy()
    if field == "axis":
        model.jnt_axis[0] = [0, 1, 0]
    elif field == "anchor":
        model.jnt_pos[0] = [0.5, 0, 0]
    elif field == "site_body":
        model.site_bodyid[0] = 0
    elif field == "active":
        model.eq_active0[0] = 1
    elif field == "coef":
        model.eq_data[0, 0] = 0.4
    else:
        model.jnt_limited[0] = 0
    after = _points(model)
    assert not np.array_equal(before, after)
    reach_workspace._CACHE.clear()
    np.testing.assert_array_equal(after, _points(model))


def test_unsupported_active_equality_does_not_claim_reachability() -> None:
    model = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody><body name="arm"><joint range="-1 1"/>
        <geom size=".1"/><site name="tip" pos="1 0 0"/></body></worldbody>
        <equality><weld body1="arm"/></equality></mujoco>"""
    )
    with pytest.raises(ValueError, match="only active joint equalities"):
        _points(model)
