# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The batched torch full-body observation builder matches the CPU reference.

``FullbodyObsAdapter`` (numpy, one ``MjData``) is the observation the MuscleMimic
checkpoints were exported against; ``TorchFullbodyObsAdapter`` builds the same
vector for a batch of envs.  Both read the same float32 state here, so they agree
to float32 rounding.  A small synthetic model keeps this independent of the
MuscleMimic assets.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mjlab")

from myosuite.core.trajectory_io import load_motion_clip  # noqa: E402
from myosuite.integrations.musclemimic.fullbody_local_policy import (  # noqa: E402
    FullbodyObsAdapter,
    _relative_site_quantities,
)
from myosuite.integrations.musclemimic.mjlab_policy_runner import (  # noqa: E402
    TorchFullbodyObsAdapter,
)

pytestmark = pytest.mark.tier1

_XML = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="pelvis" pos="0 0 1">
      <joint name="root" type="free"/>
      <geom type="box" size="0.1 0.12 0.08" mass="5"/>
      <site name="pelvis_site"/>
      <body name="thigh" pos="0.1 0 -0.1" euler="0.1 0 0.2">
        <joint name="hip_x" type="hinge" axis="1 0 0" range="-1 1"/>
        <joint name="hip_y" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.05"/>
        <site name="knee_site" pos="0 0 -0.4" euler="0.3 -0.2 0.1"/>
        <body name="shank" pos="0 0 -0.4">
          <joint name="knee" type="hinge" axis="0 1 0" range="-2 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.04"/>
          <site name="foot_site" pos="0 0 -0.4" euler="-0.4 0.2 0.6"/>
          <site name="r_foot_zone" type="sphere" size="0.1" pos="0 0 -0.4"/>
        </body>
      </body>
      <body name="arm" pos="0 0.15 0.1" euler="0.3 0.2 0.1">
        <joint name="shoulder" type="ball"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
        <site name="hand_site" pos="0.3 0 0" euler="0.5 0 0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <muscle name="m_hip_x" joint="hip_x"/>
    <muscle name="m_hip_y" joint="hip_y"/>
    <muscle name="m_knee" joint="knee"/>
  </actuator>
  <sensor>
    <touch name="r_foot" site="r_foot_zone"/>
  </sensor>
</mujoco>
"""
_SITES = ("pelvis_site", "knee_site", "foot_site", "hand_site")
_N_FRAMES = 12
_FIELDS = (
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


def _random_state(
    model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator
) -> None:
    """Tilted, spinning pose with the foot in the floor (non-zero touch)."""
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = (rng.normal(0, 0.2), rng.normal(0, 0.2), 0.85)
    axis = rng.normal(size=3)
    mujoco.mju_axisAngle2Quat(
        data.qpos[3:7], axis / np.linalg.norm(axis), rng.uniform(0.0, 0.3)
    )
    shoulder = model.jnt_qposadr[model.joint("shoulder").id]
    quat = rng.normal(size=4)
    data.qpos[shoulder : shoulder + 4] = quat / np.linalg.norm(quat)
    for name, (lo, hi) in {
        "hip_x": (-0.3, 0.3),
        "hip_y": (-0.3, 0.3),
        "knee": (-0.3, 0.0),
    }.items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = rng.uniform(lo, hi)
    data.qvel[:] = rng.normal(0.0, 1.5, model.nv)
    data.ctrl[:] = rng.uniform(0.0, 1.0, model.nu)
    data.act[:] = rng.uniform(0.0, 1.0, model.na)
    mujoco.mj_forward(model, data)


def _write_clip(model: mujoco.MjModel, path: Path) -> Path:
    """Clip NPZ in the MuscleMimic trajectory layout (site data from FK)."""
    rng = np.random.default_rng(0)
    data = mujoco.MjData(model)
    rec: dict[str, list[np.ndarray]] = {
        k: [] for k in ("qpos", "qvel", "site_xpos", "site_xmat", "cvel", "subtree_com")
    }
    for _ in range(_N_FRAMES):
        _random_state(model, data, rng)
        for key, values in rec.items():
            values.append(getattr(data, key).copy())
    np.savez(
        path,
        **{key: np.asarray(values) for key, values in rec.items()},
        site_bodyid=np.asarray(model.site_bodyid),
        body_rootid=np.asarray(model.body_rootid),
        site_names=np.asarray([model.site(i).name for i in range(model.nsite)]),
        frequency=np.asarray(100.0),
    )
    return path


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(_XML)


@pytest.fixture(scope="module")
def clip(model: mujoco.MjModel, tmp_path_factory: pytest.TempPathFactory) -> object:
    path = _write_clip(model, tmp_path_factory.mktemp("clip") / "clip.npz")
    return load_motion_clip(path, expected_nq=model.nq, expected_nv=model.nv)


def _batch(model: mujoco.MjModel, n: int, seed: int) -> list[mujoco.MjData]:
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(n):
        data = mujoco.MjData(model)
        _random_state(model, data, rng)
        states.append(data)
    return states


def _stack(states: list[mujoco.MjData]) -> SimpleNamespace:
    """Batched float32 view of the states, shaped like mjlab's sim data."""
    return SimpleNamespace(
        **{
            f: torch.as_tensor(
                np.stack([np.asarray(getattr(d, f), np.float32) for d in states])
            )
            for f in _FIELDS
        }
    )


def test_relative_angular_velocity_matches_cpu_definition() -> None:
    """Live and look-ahead site rvel use rel_rot^T w, like the CPU adapter (F1).

    With the main site at identity and the other site turned 90 deg about z,
    rel_rot^T (1, 0, 0) = (0, -1, 0) while rel_rot (1, 0, 0) = (0, 1, 0).
    """
    rz90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    site_xpos = np.array([[0.0, 0.0, 1.0], [0.3, 0.1, 0.9]], np.float32)
    site_xmat = np.stack([np.eye(3), rz90]).reshape(2, 9).astype(np.float32)
    cvel = np.array([[0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0]], np.float32)
    subtree_com = np.zeros((2, 3), np.float32)
    ids = np.array([0, 1])
    cpu = _relative_site_quantities(
        site_ids=ids,
        site_xpos=site_xpos,
        site_xmat=site_xmat,
        cvel_parent=cvel,
        subtree_com_root=subtree_com,
        site_bodyid=ids,
        body_rootid=np.zeros(2, int),
    )
    np.testing.assert_allclose(cpu[2][0, :3], [0.0, -1.0, 0.0], atol=1e-7)
    adapter = object.__new__(TorchFullbodyObsAdapter)
    torch_out = adapter._relative_site_quantities(
        site_ids=torch.as_tensor(ids),
        site_xpos=torch.as_tensor(site_xpos)[None],
        site_xmat=torch.as_tensor(site_xmat)[None],
        cvel_parent=torch.as_tensor(cvel)[None],
        subtree_com_root=torch.as_tensor(subtree_com)[None],
        site_bodyid=torch.as_tensor(ids),
        body_rootid=torch.zeros(2, dtype=torch.long),
    )
    for ref, got in zip(cpu, torch_out, strict=True):
        np.testing.assert_allclose(got[0].numpy(), ref, atol=1e-6)


@pytest.mark.parametrize("concise", [True, False], ids=["concise", "full_lookahead"])
def test_torch_adapter_matches_cpu_adapter(
    model: mujoco.MjModel, clip: object, concise: bool
) -> None:
    """Every block of the batched build equals the CPU build of each env."""
    goal = {
        "sites_for_mimic": list(_SITES),
        "n_step_lookahead": 3,
        "n_step_stride": 2,
        "use_concise_lookahead": concise,
    }
    cpu_adapter = FullbodyObsAdapter(model, clip, goal)  # type: ignore[arg-type]
    torch_adapter = TorchFullbodyObsAdapter(cpu_adapter, device=torch.device("cpu"))
    states = _batch(model, 5, seed=1)
    assert max(float(d.sensordata[0]) for d in states) > 1.0  # touch is exercised
    frames = np.array(
        [0, 3, 7, _N_FRAMES - 2, _N_FRAMES - 1]
    )  # incl. clamped look-ahead
    expected = np.stack(
        [cpu_adapter.build(d, int(f)) for d, f in zip(states, frames, strict=True)]
    )
    got = torch_adapter.build(_stack(states), torch.as_tensor(frames)).numpy()
    assert got.shape == expected.shape and got.dtype == np.float32
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
