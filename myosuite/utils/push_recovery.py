# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Paired headless native push experiments; no training algorithm or renderer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from myosuite.envs.wrappers import PerturbationWrapper
from myosuite.utils.push_protocol import PushProtocol
from myosuite.utils.paired_replay import ReplayPolicy, ReplaySnapshot, ReplayStateOwner
from myosuite.utils.push_scheduling import ContactPhase, jitter_free_root


@dataclass
class PushTrace:
    """Post-step states include t=0; actions/rewards/wrenches index transitions."""

    states: dict[str, np.ndarray]
    actions: np.ndarray
    observations: np.ndarray
    rewards: np.ndarray
    applied_wrench: np.ndarray
    terminated: bool
    truncated: bool


@dataclass
class PairedPushTrial:
    """One seed and condition, with the common saved initial state."""

    seed: int
    protocol: dict[str, Any]
    baseline: PushTrace
    pushed: PushTrace
    initial_integration_state: np.ndarray
    initial_controller_state: Any
    initial_rng_state: dict[str, Any]


def _sample(
    model: mujoco.MjModel, data: mujoco.MjData, resolved: dict
) -> dict[str, Any]:
    root = resolved["root_id"]
    sample = {
        "time_s": float(data.time),
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "act": data.act.copy(),
        "com_m": data.subtree_com[root].copy(),
        "root_rotation": data.xmat[root].reshape(3, 3).copy(),
        "muscle_excitation": data.ctrl[resolved["muscle_actuator_ids"]].copy(),
        "muscle_activation": data.act[resolved["muscle_state_indices"]].copy(),
        "muscle_force_n": data.actuator_force[resolved["muscle_actuator_ids"]].copy(),
    }
    if resolved["foot_body_ids"]:
        count, vertical = 0, 0.0
        mask = np.zeros(len(resolved["foot_body_ids"]), dtype=bool)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            if contact.geom1 < 0 or contact.geom2 < 0:
                continue
            first, second = model.geom_bodyid[[contact.geom1, contact.geom2]]
            foot_first = (
                first in resolved["foot_body_ids"]
                and second in resolved["ground_body_ids"]
            )
            foot_second = (
                second in resolved["foot_body_ids"]
                and first in resolved["ground_body_ids"]
            )
            if not (foot_first or foot_second):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, contact_id, force)
            sign = 1 if foot_second else -1
            vertical += sign * float((contact.frame.reshape(3, 3).T @ force[:3])[2])
            if force[0] > resolved["contact_threshold_n"]:
                count += 1
                mask[
                    resolved["foot_body_ids"].index(
                        int(second if foot_second else first)
                    )
                ] = True
        sample.update(
            foot_contact_count=count,
            vertical_grf_n=vertical,
            foot_contact_mask=mask,
            foot_positions_m=data.xpos[resolved["foot_body_ids"]].copy(),
        )
    return sample


def _rollout(
    env: PerturbationWrapper,
    policy: ReplayPolicy,
    obs: np.ndarray,
    resolved: dict,
    steps: int,
) -> PushTrace:
    u = env.unwrapped
    samples, actions, observations, rewards, wrenches = (
        [_sample(u.model, u.data, resolved)],
        [],
        [obs.copy()],
        [],
        [],
    )
    terminated = truncated = False
    for step in range(steps):
        action = np.asarray(policy.action(obs), dtype=np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        obs = np.asarray(obs, dtype=np.float32)
        samples.append(_sample(u.model, u.data, resolved))
        if (
            abs(
                samples[-1]["time_s"]
                - samples[0]["time_s"]
                - (step + 1) * resolved["ctrl_dt_s"]
            )
            > 1e-8
        ):
            raise ValueError(
                "Env physics clock does not match the declared control step"
            )
        actions.append(action.copy())
        observations.append(obs.copy())
        rewards.append(float(reward))
        wrenches.append(u.data.xfrc_applied[resolved["body_id"]].copy())
        if terminated or truncated:
            break
    return PushTrace(
        {key: np.asarray([s[key] for s in samples]) for key in samples[0]},
        np.asarray(actions),
        np.asarray(observations),
        np.asarray(rewards),
        np.asarray(wrenches),
        bool(terminated),
        bool(truncated),
    )


def run_paired_push(
    env: gym.Env,
    policy: ReplayPolicy,
    protocol: PushProtocol,
    *,
    seed: int,
    state_owners: tuple[ReplayStateOwner, ...] = (),
    contact_phase: ContactPhase | None = None,
    velocity_jitter_natural: float = 0.0,
) -> PairedPushTrial:
    """Reset, replay both arms from one snapshot and restore it even after errors.

    The env seed controls reset; controller-owned/global RNG initial states are
    explicitly captured, not guessed from it. This function leaves the initialized
    env/controller at the captured reset state. No existing perturbation wrapper
    may be present. Missing controller state is detected when pre-push traces differ.
    """
    layer = env
    while isinstance(layer, gym.Wrapper):
        if isinstance(layer, PerturbationWrapper):
            raise ValueError("Pass an env without an existing PerturbationWrapper")
        layer = layer.env
    wrapped = PerturbationWrapper(env)
    obs, _ = env.reset(seed=seed)
    if not isinstance(obs, np.ndarray) or obs.ndim != 1:
        raise ValueError("The paired protocol requires flat CPU observations")
    resolved = protocol.resolve(wrapped)
    if contact_phase is not None and protocol.phase is not None:
        raise ValueError("Use measured contact phase or nominal clock phase, not both")
    obs, resolved = jitter_free_root(
        env, obs, resolved, seed=seed, std_natural=velocity_jitter_natural
    )
    if np.any(wrapped.unwrapped.data.xfrc_applied[resolved["body_id"]]):
        raise ValueError(
            "Paired pushes require zero background wrench on the target body"
        )
    snapshot = ReplaySnapshot.capture(wrapped, policy, state_owners=state_owners)
    try:
        baseline = _rollout(wrapped, policy, obs.copy(), resolved, protocol.steps)
        if contact_phase is not None:
            resolved = contact_phase.schedule(baseline.states, resolved)
        snapshot.restore()
        if resolved.get("phase_available", True):
            wrapped.add_perturbation(
                {
                    "body": protocol.body,
                    "force": resolved["force_world_or_body_n"],
                    "frame": protocol.frame,
                    "point": protocol.point_body_m,
                    "start": resolved["start_step"],
                    "end": resolved["end_step"],
                }
            )
        pushed = _rollout(wrapped, policy, obs.copy(), resolved, protocol.steps)
        n = resolved["start_step"]
        for key in ("qpos", "qvel", "act"):
            if not np.array_equal(
                baseline.states[key][: n + 1], pushed.states[key][: n + 1]
            ):
                raise ValueError(
                    "Paired pre-push physics differs; restore all controller/env state"
                )
        if not np.array_equal(
            baseline.actions[:n], pushed.actions[:n]
        ) or not np.array_equal(
            baseline.observations[: n + 1], pushed.observations[: n + 1]
        ):
            raise ValueError(
                "Paired pre-push observations/actions differ; restore controller history/RNG"
            )
        return PairedPushTrial(
            seed,
            resolved,
            baseline,
            pushed,
            snapshot.integration_state(),
            snapshot.policy_state,
            snapshot.rng_state(),
        )
    finally:
        snapshot.restore()
