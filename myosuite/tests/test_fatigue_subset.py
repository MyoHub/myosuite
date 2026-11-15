"""3CC-r fatigue on a subset of the muscles (``muscles=``)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.core.muscle_conditions import CumulativeFatigue, TorchFatigueState
from myosuite.envs.wrappers import FatigueWrapper

pytestmark = pytest.mark.tier1

_SUBSET_ENV = "myoElbowPose1D6MRandom-v0"


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    env = make_env("myoChallengeBimanual-v0")  # muscles plus prosthesis motors
    yield env.unwrapped.model
    env.close()


def _muscles(model: mujoco.MjModel) -> np.ndarray:
    return np.flatnonzero(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)


def test_default_is_every_muscle(model: mujoco.MjModel) -> None:
    np.testing.assert_array_equal(CumulativeFatigue(model).muscle_ids, _muscles(model))


def test_subset_follows_the_full_model(model: mujoco.MjModel) -> None:
    """A subset (by name, any order) evolves exactly as those rows of the full model."""
    muscles = _muscles(model)
    rows = [7, 2, 40]
    names = [model.actuator(int(muscles[i])).name for i in rows]
    full = CumulativeFatigue(model, frame_skip=10)
    sub = CumulativeFatigue(model, frame_skip=10, muscles=names)
    np.testing.assert_array_equal(sub.muscle_ids, muscles[rows])
    for name in ("F", "R", "r"):
        np.testing.assert_array_equal(getattr(sub, name), getattr(full, name)[rows])
    rng = np.random.default_rng(0)
    for _ in range(300):
        exc = rng.uniform(0.0, 1.0, full.na)
        full.compute_act(exc)
        sub.compute_act(exc[rows])
    for comp in ("MA", "MF", "MR"):
        np.testing.assert_allclose(getattr(sub, comp), getattr(full, comp)[rows])
    assert sub.MF.min() > 0.0


def test_subset_rejects_non_muscles(model: mujoco.MjModel) -> None:
    motor = model.actuator("prosthesis/Lelbow").id
    with pytest.raises(ValueError, match="not muscle actuators"):
        CumulativeFatigue(model, muscles=[motor])


def test_torch_subset_matches_numpy(model: mujoco.MjModel) -> None:
    pytest.importorskip("torch")
    ids = _muscles(model)[[3, 40, 11]]
    cpu = CumulativeFatigue(model, muscles=ids)
    gpu = TorchFatigueState.from_mj_model(model, num_envs=2, muscles=ids)
    for name, ours in (("F", gpu._F), ("R", gpu._R), ("r", gpu._r)):
        np.testing.assert_allclose(ours.numpy(), getattr(cpu, name), rtol=1e-6)
    assert gpu.MF.shape == (2, 3)


def test_wrapper_fatigues_only_the_given_muscles() -> None:
    """Listed muscles fatigue; the others get the same ctrl as without fatigue."""
    plain = make_env(_SUBSET_ENV)
    env = make_env(_SUBSET_ENV)
    u = env.unwrapped
    listed = [0, 3]
    names = [u.model.actuator(i).name for i in listed]
    env = FatigueWrapper(env, muscles=names)
    plain.reset(seed=0)
    env.reset(seed=0)
    action = np.ones(u.model.nu)
    for _ in range(200):
        plain.step(action)
        env.step(action)
    assert env.muscle_fatigue.MF.shape == (2,) and env.muscle_fatigue.MF.min() > 0.0
    others = np.setdiff1d(np.arange(u.model.nu), listed)
    np.testing.assert_allclose(u.data.ctrl[others], plain.unwrapped.data.ctrl[others])
    assert (u.data.ctrl[listed] < plain.unwrapped.data.ctrl[listed]).all()
    env.close()
    plain.close()
