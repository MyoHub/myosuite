"""No reach reset starts beyond the far threshold, so a far termination is the policy's doing.

``multi_site_reach_reward`` ends an episode once the tip-target distance (norm over all
``k`` sites) exceeds ``far_th * k``, from the second control step on. A reset already
beyond it ends the episode at step 2 unless the policy closes the gap within two steps.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import myosuite  # noqa: F401

_REACH_EP = "tasks.basic.arm.reach:ReachEnvV0"
_N_SEEDS = 20


def _reach_ids() -> list[str]:
    return sorted(
        env_id
        for env_id, spec in gym.registry.items()
        if str(spec.entry_point).endswith(_REACH_EP)
    )


def _farthest_target(u: gym.Env, tip: np.ndarray) -> float:
    """Largest tip-target distance (norm over all sites) the target sampler can produce."""
    if u._workspace_points is not None:
        offsets = (u._workspace_points - tip).reshape(len(u._workspace_points), -1)
        return float(np.linalg.norm(offsets, axis=1).max())
    low, high = (
        np.array([span[i] for span in u.target_reach_range.values()], dtype=float)
        for i in (0, 1)
    )
    # Each coordinate is drawn independently, so the farthest target is a box corner.
    return float(np.linalg.norm(np.maximum(np.abs(low - tip), np.abs(high - tip))))


@pytest.mark.tier1
@pytest.mark.parametrize("env_id", _reach_ids())
def test_no_reset_starts_beyond_far_threshold(env_id: str) -> None:
    env = gym.make(env_id)
    u = env.unwrapped
    far = u.far_th * len(u.tip_sids)
    env.reset(seed=0)
    start_tip = u.data.site_xpos[u.tip_sids].copy()
    farthest = _farthest_target(u, start_tip)
    assert (
        farthest < far
    ), f"{env_id}: targets up to {farthest:.3f} m from the start tips, far threshold {far:.3f} m"
    for seed in range(_N_SEEDS):
        env.reset(seed=seed)
        tip = u.data.site_xpos[u.tip_sids]
        # The bound above holds because every reset starts from the same pose.
        np.testing.assert_allclose(tip, start_tip, atol=1e-9)
        target = u.data.site_xpos[u.target_sids]
        assert np.linalg.norm(target - tip) < far
    env.close()


@pytest.mark.tier2
@pytest.mark.parametrize(
    "env_id", ["myoArmReachRandom-v0", "myoHandReachFixed-v0", "myoHandReachRandom-v0"]
)
def test_mjlab_twin_resets_within_far_threshold(env_id: str) -> None:
    pytest.importorskip("mjlab")
    torch = pytest.importorskip("torch")
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import myosuite.envs.myo.backends.mjlab  # noqa: F401 (registers the twins)

    far_th = gym.spec(env_id).kwargs["far_th"]
    cfg = load_env_cfg(env_id)
    assert cfg.terminations["reach_failed"].params["far_th"] == far_th
    cfg.scene.num_envs = 256
    torch.manual_seed(0)
    mj = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    mj.reset()
    command = mj.command_manager.get_term("reach")
    tip = command._accessor.site_xpos(command._tip_ids).reshape(mj.num_envs, -1)
    dist = torch.linalg.norm(command.command - tip, dim=-1)
    assert float(dist.max()) < far_th * len(command._tip_ids)
    mj.close()
