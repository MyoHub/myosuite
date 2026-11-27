# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Batched GPU mjlab (MuJoCo Warp + PyTorch) driver and trainer for DreamerV3."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import os
from pathlib import Path
import sys
import time
from typing import Any

import elements
import numpy as np

from myosuite_dreamerv3.compat import import_myosuite, resolve_jax_platform
from myosuite_dreamerv3.export import (
    export_dreamerv3_onnx,
    export_dreamerv3_torchscript,
)
from myosuite_dreamerv3.train import (
    build_dreamer_config,
    evaluate_dreamer_agent,
    make_agent_and_factories,
    write_policy_metadata,
)
from myosuite_dreamerv3.train_mjx import run_mjx_training_loop

# Sanity bounds for observations / rewards of diverged Warp worlds (see _sanitize).
OBS_BOUND = 1e4
REWARD_BOUND = 1e4


def _extract_actor_obs(obs_dict: Any) -> Any:
    """Extract policy observation tensor from an ``mjlab`` observation mapping."""
    if isinstance(obs_dict, dict):
        if "actor" in obs_dict:
            return obs_dict["actor"]
        if "policy" in obs_dict:
            return obs_dict["policy"]
        return next(iter(obs_dict.values()))
    return obs_dict


def mjlab_action_bounds(env: Any) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension normalized action bounds of an ``mjlab`` env's action terms.

    DreamerV3 always acts in ``[-1, 1]``. On the CPU backend, ``NormalizeAction``
    maps that range linearly onto the env's ``action_space`` (``[0, 1]`` for the
    leg-walk envs, ``[-1, 1]`` for pose/reach). The ``mjlab`` twins instead *clamp*
    raw actions to their term's ``action_range``, so without the same linear map the
    negative half of DreamerV3's range collapses to zero activation on the walk twins
    and a trained policy sees different muscle commands on CPU than in training.

    Args:
        env: Instantiated ``ManagerBasedRlEnv``.

    Returns:
        ``(low, high)`` float32 arrays of length ``total_action_dim``.
    """
    am = env.action_manager
    dim = int(am.total_action_dim)
    low = np.full(dim, -1.0, np.float32)
    high = np.full(dim, 1.0, np.float32)
    col = 0
    for name in getattr(am, "active_terms", ()):
        term = am.get_term(name)
        d = int(term.action_dim)
        cfg = getattr(term, "cfg", None)
        rng = getattr(cfg, "action_range", None)
        if rng is not None:
            if getattr(cfg, "normalize_act", True):
                low[col : col + d], high[col : col + d] = float(rng[0]), float(rng[1])
            elif hasattr(term, "_ctrl_lo") and hasattr(term, "_ctrl_hi"):
                low[col : col + d] = term._ctrl_lo.detach().cpu().numpy()
                high[col : col + d] = term._ctrl_hi.detach().cpu().numpy()
        col += d
    return low, high


def load_myosuite_mjlab_env(
    env_id: str,
    num_envs: int = 64,
    seed: int = 0,
    device: str = "cuda:0",
    features: Sequence[Any] = (),
) -> Any:
    """Instantiate a registered MyoSuite ``mjlab`` GPU environment twin.

    Args:
        env_id: Registered MyoSuite environment ID (e.g. ``myoElbowPose1D6MRandom-v0``).
        num_envs: Number of parallel environments on device.
        seed: Environment random seed.
        device: PyTorch / Warp device string (``"cuda:0"`` or ``"cpu"``).
        features: Optional sequence of ``WrapperSpec`` or ``--feature`` strings.

    Returns:
        Instantiated ``ManagerBasedRlEnv``.
    """
    import_myosuite()
    import myosuite.envs.myo.backends.mjlab  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    os.environ.setdefault("MUJOCO_GL", "egl")
    if features:
        from myosuite.envs.myo.backends.mjlab.tasks.registration import rebuild_twin_cfg
        from myosuite.utils.feature_cli import feature_spec

        parsed = tuple(feature_spec(f) if isinstance(f, str) else f for f in features)
        env_cfg = rebuild_twin_cfg(env_id, parsed)
    else:
        env_cfg = load_env_cfg(env_id)

    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.seed = int(seed)
    return ManagerBasedRlEnv(cfg=env_cfg, device=device)


class MjlabDriver:
    """Batched GPU driver connecting MyoSuite's ``mjlab`` backend to DreamerV3.

    Handles ``ManagerBasedRlEnv``'s same-step auto-reset contract so DreamerV3's
    replay buffer and recurrent RSSM receive chronological ``is_first=True`` and
    ``is_last=True`` transitions without skipping initial episode observations.

    Args:
        env_or_ctor: Instantiated ``ManagerBasedRlEnv`` or zero-arg constructor.
        act_key: Key of the continuous action tensor in ``act_space``.
        **kwargs: Forwarded to policy and step callbacks.
    """

    def __init__(
        self,
        env_or_ctor: Any,
        act_key: str = "action",
        **kwargs: Any,
    ) -> None:
        import torch

        self._torch = torch
        self.env = env_or_ctor() if callable(env_or_ctor) else env_or_ctor
        self.length = int(self.env.num_envs)
        self.device = getattr(self.env, "device", "cpu")
        self.act_key = act_key
        self.kwargs = kwargs
        self.sanitized_values = 0
        self._has_success = hasattr(
            self.env, "metrics_manager"
        ) and "success" in getattr(self.env.metrics_manager, "active_terms", ())

        with torch.no_grad():
            init_obs_raw, _ = self.env.reset()
            init_obs = _extract_actor_obs(init_obs_raw).detach().cpu().numpy()

        self._obs_shape = tuple(init_obs.shape[1:])
        act_dim = int(self.env.action_manager.total_action_dim)
        self._act_low, self._act_high = mjlab_action_bounds(self.env)

        self.obs_space: dict[str, elements.Space] = {
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
            "proprio": elements.Space(np.float32, self._obs_shape),
            "log/solved": elements.Space(np.float32),
        }
        self.act_space: dict[str, elements.Space] = {
            self.act_key: elements.Space(np.float32, (act_dim,), -1.0, 1.0),
            "reset": elements.Space(bool),
        }

        self.callbacks: list[Callable[..., Any]] = []
        self.acts: dict[str, np.ndarray] = {}
        self.carry: Any = None
        self._pending_first_obs: np.ndarray = init_obs.astype(np.float32, copy=False)
        self._last_obs: np.ndarray = self._pending_first_obs.copy()
        self._needs_initial_emit = True
        self.reset()

    def reset(self, init_policy: Callable[[int], Any] | None = None) -> None:
        """Reset driver action state and initialize policy recurrent carry."""
        torch = self._torch
        with torch.no_grad():
            obs_raw, _ = self.env.reset()
            self._pending_first_obs = (
                _extract_actor_obs(obs_raw).detach().cpu().numpy().astype(np.float32)
            )
            self._last_obs = self._pending_first_obs.copy()
        self._needs_initial_emit = True
        self.acts = {
            k: np.zeros((self.length, *v.shape), v.dtype)
            for k, v in self.act_space.items()
        }
        self.carry = init_policy(self.length) if init_policy is not None else None

    def close(self) -> None:
        """Close the underlying ``mjlab`` environment."""
        self.env.close()

    def _sanitize(self, x: np.ndarray, bound: float) -> np.ndarray:
        """Replace non-finite values and clip to ``[-bound, bound]``.

        A diverged MuJoCo Warp world can emit NaN/inf or huge values for a step; one
        such transition in the replay buffer turns every DreamerV3 parameter into NaN.
        Bounds sit far outside the values of a regular step, so they only touch
        diverged worlds.
        """
        bad = ~np.isfinite(x) | (np.abs(x) > bound)
        if bad.any():
            self.sanitized_values += int(bad.sum())
            x = np.clip(
                np.nan_to_num(x, nan=0.0, posinf=bound, neginf=-bound), -bound, bound
            )
        return x

    def on_step(self, callback: Callable[..., Any]) -> None:
        """Register a per-transition callback ``fn(tran, worker_idx)``."""
        self.callbacks.append(callback)

    def __call__(
        self, policy: Callable[..., Any], steps: int = 0, episodes: int = 0
    ) -> None:
        """Roll out ``policy`` for at least ``steps`` transitions or ``episodes``."""
        step, episode = 0, 0
        while step < steps or episode < episodes:
            step, episode = self._step(policy, step, episode)

    def _emit_batch(
        self,
        policy: Callable[..., Any],
        obs_proprio: np.ndarray,
        rew: np.ndarray,
        is_first: np.ndarray,
        is_last: np.ndarray,
        is_term: np.ndarray,
        solved: np.ndarray,
        active_indices: Sequence[int] | None = None,
    ) -> dict[str, np.ndarray]:
        obs = {
            "proprio": obs_proprio.astype(np.float32, copy=False),
            "reward": rew.astype(np.float32, copy=False),
            "is_first": is_first.astype(bool, copy=False),
            "is_last": is_last.astype(bool, copy=False),
            "is_terminal": is_term.astype(bool, copy=False),
        }
        logs = {"log/solved": solved.astype(np.float32, copy=False)}
        prev_carry = self.carry
        new_carry, acts, outs = policy(self.carry, obs, **self.kwargs)

        if active_indices is None:
            self.carry = new_carry
            indices = range(self.length)
        else:
            active_mask = np.zeros(self.length, dtype=bool)
            active_mask[list(active_indices)] = True
            if prev_carry is not None and new_carry is not None:

                def is_env_list(x):
                    return isinstance(x, list) and len(x) == self.length

                def _select_carry(old: Any, new: Any) -> Any:
                    if is_env_list(new):
                        return [
                            new[idx] if active_mask[idx] else old[idx]
                            for idx in range(self.length)
                        ]
                    return np.where(
                        active_mask.reshape((-1,) + (1,) * (new.ndim - 1)), new, old
                    )

                self.carry = elements.tree.map(
                    _select_carry,
                    prev_carry,
                    new_carry,
                    isleaf=is_env_list,
                )
            else:
                self.carry = new_carry
            indices = active_indices  # type: ignore[assignment]

        trans = {**obs, **acts, **outs, **logs}
        for i in indices:
            trn = elements.tree.map(lambda x, idx=i: x[idx], trans)
            for fn in self.callbacks:
                fn(trn, i, **self.kwargs)
        return acts

    def _step(
        self, policy: Callable[..., Any], step: int, episode: int
    ) -> tuple[int, int]:
        torch = self._torch
        n = self.length

        if self._needs_initial_emit:
            self._needs_initial_emit = False
            acts = self._emit_batch(
                policy,
                obs_proprio=self._pending_first_obs,
                rew=np.zeros(n, dtype=np.float32),
                is_first=np.ones(n, dtype=bool),
                is_last=np.zeros(n, dtype=bool),
                is_term=np.zeros(n, dtype=bool),
                solved=np.zeros(n, dtype=np.float32),
            )
            self.acts = {**acts, "reset": np.zeros(n, dtype=bool)}
            self._last_obs = self._pending_first_obs.copy()
            return step + n, episode

        act_np = np.clip(
            np.asarray(self.acts[self.act_key], dtype=np.float32), -1.0, 1.0
        )
        # Same linear [-1, 1] -> action_space map as the CPU NormalizeAction wrapper.
        act_np = self._act_low + (act_np + 1.0) * 0.5 * (self._act_high - self._act_low)
        with torch.no_grad():
            act_tensor = torch.as_tensor(
                act_np, dtype=torch.float32, device=self.device
            )
            obs_raw, rew_t, term_t, trunc_t, extras = self.env.step(act_tensor)
            next_obs = self._sanitize(
                _extract_actor_obs(obs_raw).detach().cpu().numpy().astype(np.float32),
                OBS_BOUND,
            )
            rew = self._sanitize(
                rew_t.detach().cpu().numpy().astype(np.float32), REWARD_BOUND
            )
            terminated = term_t.detach().cpu().numpy().astype(bool)
            truncated = trunc_t.detach().cpu().numpy().astype(bool)
            done = terminated | truncated

            solved = np.zeros(n, dtype=np.float32)
            if self._has_success and done.any():
                for idx in np.flatnonzero(done).tolist():
                    terms = dict(
                        self.env.metrics_manager.get_active_iterable_terms(idx)
                    )
                    if "success" in terms and terms["success"]:
                        solved[idx] = float(terms["success"][0])

        term_obs = next_obs.copy()
        if done.any():
            extra_obs = (
                extras.get("observations", {}) if isinstance(extras, dict) else {}
            )
            if isinstance(extra_obs, dict) and "terminal" in extra_obs:
                term_obs[done] = self._sanitize(
                    extra_obs["terminal"][done]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                    OBS_BOUND,
                )
            else:
                term_obs[done] = self._last_obs[done]

        acts = self._emit_batch(
            policy,
            obs_proprio=term_obs,
            rew=rew,
            is_first=np.zeros(n, dtype=bool),
            is_last=done,
            is_term=terminated,
            solved=solved,
        )
        step += n
        episode += int(done.sum())

        if done.any():
            done_indices = np.flatnonzero(done).tolist()
            reset_acts = self._emit_batch(
                policy,
                obs_proprio=next_obs,
                rew=np.zeros(n, dtype=np.float32),
                is_first=done,
                is_last=np.zeros(n, dtype=bool),
                is_term=np.zeros(n, dtype=bool),
                solved=np.zeros(n, dtype=np.float32),
                active_indices=done_indices,
            )
            for k in acts:
                acts[k] = acts[k].copy()
                acts[k][done] = reset_acts[k][done]
            step += len(done_indices)

        self._last_obs = next_obs.copy()
        self.acts = {**acts, "reset": np.zeros(n, dtype=bool)}
        return step, episode


def train_mjlab(
    env_id: str = "myoElbowPose1D6MRandom-v0",
    num_envs: int = 64,
    steps: int = 200_000,
    train_ratio: float = 128.0,
    size: str = "size12m",
    platform: str = "cuda",
    sim_device: str = "cuda:0",
    seed: int = 0,
    logdir: str = "",
    export_onnx: str = "",
    export_torchscript: str = "",
    outputs: Sequence[str] = ("jsonl",),
    features: Sequence[Any] = (),
    eval_episodes: int = 0,
    curve_steps: Sequence[int] = (),
    debug: bool = False,
) -> dict[str, Any]:
    """Train DreamerV3 on a batched GPU MyoSuite ``mjlab`` twin environment."""
    platform = resolve_jax_platform(platform)
    resolved_logdir = logdir or str(
        Path("dreamerv3_mjlab_logs")
        / f"{env_id}_{num_envs}_seed{seed}_{int(time.time())}"
    )
    config = build_dreamer_config(
        size=size,
        platform=platform,
        seed=seed,
        steps=steps,
        envs=num_envs,
        train_ratio=train_ratio,
        logdir=resolved_logdir,
        outputs=outputs,
        debug=debug,
    ).update({"jax.prealloc": False})
    elements.Path(config.logdir).mkdir()
    config.save(elements.Path(config.logdir) / "config.yaml")

    mjlab_env = load_myosuite_mjlab_env(
        env_id,
        num_envs=num_envs,
        seed=seed,
        device="cpu" if debug and sim_device.startswith("cuda") else sim_device,
        features=features,
    )
    driver = MjlabDriver(mjlab_env)
    write_policy_metadata(
        config.logdir,
        config,
        driver.obs_space,
        driver.act_space,
        env_id=env_id,
    )
    agent, make_replay, make_stream, make_logger = make_agent_and_factories(
        config, driver.obs_space, driver.act_space
    )
    eval_cb = (
        (
            lambda ag: evaluate_dreamer_agent(
                ag,
                env_id=env_id,
                episodes=max(1, eval_episodes),
                seed=seed + 10_000,
                features=features,
            )
        )
        if eval_episodes > 0 and curve_steps
        else None
    )
    summary = run_mjx_training_loop(
        driver,  # type: ignore[arg-type]
        agent,
        make_replay,
        make_stream,
        make_logger,
        config,
        eval_fn=eval_cb,
        curve_steps=curve_steps,
    )
    summary.update({"env_id": env_id, "num_envs": num_envs, "seed": seed})
    if eval_episodes > 0:
        curve = summary.get("curve", [])
        if curve and curve[-1]["steps"] == int(steps):
            summary["eval"] = {
                k: curve[-1][k]
                for k in ("mean_return", "std_return", "solved_rate", "mean_length")
            }
        else:
            summary["eval"] = evaluate_dreamer_agent(
                agent,
                env_id=env_id,
                episodes=eval_episodes,
                seed=seed + 10_000,
                features=features,
            )
    best_dir = Path(str(config.logdir)) / "best_ckpt"
    export_src = best_dir if (best_dir / "agent.pkl").is_file() else config.logdir
    if export_torchscript:
        export_dreamerv3_torchscript(export_src, export_torchscript)
        summary["torchscript"] = str(Path(export_torchscript).resolve())
    if export_onnx:
        export_dreamerv3_onnx(export_src, export_onnx, env_id=env_id, features=features)
        summary["onnx"] = str(Path(export_onnx).resolve())
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """CLI entrypoint for ``dreamerv3-train-mjlab``."""
    from myosuite.utils.feature_cli import parse_feature_args

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    features, rest = parse_feature_args(raw_argv)

    parser = argparse.ArgumentParser(
        description="Train DreamerV3 on batched GPU MyoSuite mjlab environments."
    )
    parser.add_argument("--env-id", type=str, default="myoElbowPose1D6MRandom-v0")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--train-ratio", type=float, default=128.0)
    parser.add_argument(
        "--size",
        type=str,
        default="size12m",
        choices=("size1m", "size12m", "size25m", "size50m"),
    )
    parser.add_argument(
        "--platform", type=str, default="cuda", choices=("cuda", "cpu", "tpu")
    )
    parser.add_argument("--sim-device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="")
    parser.add_argument("--export-onnx", type=str, default="")
    parser.add_argument("--export-torchscript", type=str, default="")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(rest)

    result = train_mjlab(
        env_id=args.env_id,
        num_envs=args.num_envs,
        steps=args.steps,
        train_ratio=args.train_ratio,
        size=args.size,
        platform=args.platform,
        sim_device=args.sim_device,
        seed=args.seed,
        logdir=args.logdir,
        export_onnx=args.export_onnx,
        export_torchscript=args.export_torchscript,
        features=features,
        eval_episodes=args.eval_episodes,
        debug=args.debug,
    )
    print(
        f"Completed {result['steps']} steps on {args.env_id} (mjlab, "
        f"{args.num_envs} envs) in {result['wall_time_s']:.2f}s."
    )
    return result


if __name__ == "__main__":
    main()
