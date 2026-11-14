"""CPU-layout state writes retain entity isolation and free-joint conventions."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

from myosuite import make_env  # noqa: E402
from myosuite.envs.myo.backends.mjlab.tasks.mdp.events import write_cpu_state  # noqa: E402


@pytest.mark.parametrize(
    "env_id", ["myoElbowPose1D6MFixed-v0", "myoLegDirectionalForward-v0"]
)
def test_cpu_state_roundtrip_with_selected_environments(env_id: str) -> None:
    env = make_env(env_id, backend="mjlab", device="cpu", num_envs=2)
    try:
        model = env.sim.mj_model
        before_q = env.sim.data.qpos.clone()
        before_v = env.sim.data.qvel.clone()
        q = before_q[[1]].clone()
        v = torch.linspace(-0.2, 0.3, model.nv)[None]
        if not env.scene["robot"].is_fixed_base:
            q[:, 3:7] = torch.tensor([0.5, 0.5, 0.5, 0.5])
            q[:, :3] -= env.scene.env_origins[[1]]
        original_q, original_v = q.clone(), v.clone()
        write_cpu_state(env, "robot", torch.tensor([1]), q, v)
        expected_q = original_q.clone()
        if not env.scene["robot"].is_fixed_base:
            expected_q[:, :3] += env.scene.env_origins[[1]]
        np.testing.assert_allclose(
            env.sim.data.qpos[[1]].numpy(), expected_q.numpy(), atol=1e-7
        )
        np.testing.assert_allclose(env.sim.data.qvel[[1]].numpy(), v.numpy(), atol=1e-7)
        torch.testing.assert_close(
            env.sim.data.qpos[[0]], before_q[[0]], rtol=0, atol=0
        )
        torch.testing.assert_close(
            env.sim.data.qvel[[0]], before_v[[0]], rtol=0, atol=0
        )
        torch.testing.assert_close(q, original_q, rtol=0, atol=0)
        torch.testing.assert_close(v, original_v, rtol=0, atol=0)
    finally:
        env.close()
