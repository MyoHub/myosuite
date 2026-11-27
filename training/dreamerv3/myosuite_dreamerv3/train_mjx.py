# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Train DreamerV3 on GPU-batched MyoSuite MJX / Warp environments."""

from __future__ import annotations

import argparse
import collections
from collections.abc import Callable, Sequence
from pathlib import Path
import time
from typing import Any

import elements
import embodied

from myosuite_dreamerv3.compat import import_myosuite, resolve_jax_platform
from myosuite_dreamerv3.mjx_driver import MJXDriver
from myosuite_dreamerv3.train import (
    build_dreamer_config,
    make_agent_and_factories,
    write_policy_metadata,
)


def load_myosuite_mjx_env(
    env_name: str,
    num_envs: int = 64,
    impl: str = "jax",
) -> Any:
    """Instantiate a MyoSuite MJX environment across MyoSuite 3.x and 2.x layouts.

    Args:
        env_name: Registered MJX task name (e.g. ``MjxElbowPoseRandom-v0``).
        num_envs: Parallel environments on device.
        impl: MJX physics backend (``"jax"`` or ``"warp"``).

    Returns:
        Instantiated MJX environment.
    """
    import_myosuite()
    if impl == "warp":
        try:
            import mujoco.mjx.warp as mjxw

            if mjxw.types.GraphMode is int:
                import warnings

                warnings.warn(
                    "mujoco.mjx Warp FFI bridge is unavailable with the installed "
                    "warp-lang version; falling back to impl='jax' (use "
                    "dreamerv3-train-mjlab for native mujoco-warp GPU simulation).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                impl = "jax"
        except Exception:
            impl = "jax"
    overrides: dict[str, Any] = {"num_envs": int(num_envs)}
    try:
        from myosuite.envs.myo.backends import mjx as myo_mjx

        cfg = myo_mjx.get_default_config(env_name)
        if hasattr(cfg, "mjx_impl"):
            overrides["mjx_impl"] = None if impl in ("jax", "", None) else impl
        else:
            overrides["impl"] = impl
        return myo_mjx.make(env_name, config_overrides=overrides)
    except ImportError:
        from myosuite.envs.myo.mjx import make as mjx_make

        overrides["impl"] = impl
        return mjx_make(env_name, config_overrides=overrides)


def run_mjx_training_loop(
    driver: Any,
    agent: Any,
    make_replay: Any,
    make_stream: Any,
    make_logger: Any,
    config: elements.Config,
    *,
    eval_fn: Callable[[Any], dict[str, float]] | None = None,
    curve_steps: Sequence[int] = (),
) -> dict[str, Any]:
    """Execute DreamerV3 world-model training driven by a batched driver."""
    replay = make_replay("replay", "train")
    logger = make_logger()
    logdir = elements.Path(config.logdir)
    step = logger.step

    train_agg = elements.Agg()
    epstats = elements.Agg()
    episodes: dict[int, elements.Agg] = collections.defaultdict(elements.Agg)
    policy_fps = elements.FPS()
    train_fps = elements.FPS()

    batch_steps = config.batch_size * config.batch_length
    should_train = elements.when.Ratio(config.run.train_ratio / batch_steps)
    should_log = embodied.LocalClock(config.run.log_every)
    should_report = embodied.LocalClock(config.run.report_every)
    should_save = embodied.LocalClock(config.run.save_every)

    def logfn(tran: dict[str, Any], worker: int) -> None:
        episode = episodes[worker]
        if tran["is_first"]:
            episode.reset()
        episode.add("score", tran["reward"], agg="sum")
        episode.add("length", 1, agg="sum")
        for key, value in tran.items():
            if key.startswith("log/"):
                episode.add(f"{key}/avg", value, agg="avg")
        if tran["is_last"]:
            result = episode.result()
            logger.add(
                {"score": result.pop("score"), "length": result.pop("length")},
                prefix="episode",
            )
            epstats.add(result)

    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(lambda tran, _: policy_fps.step())
    driver.on_step(replay.add)
    driver.on_step(logfn)

    stream_train = iter(agent.stream(make_stream(replay, "train")))
    stream_report = iter(agent.stream(make_stream(replay, "report")))
    carry_train = [agent.init_train(config.batch_size)]
    carry_report = agent.init_report(config.batch_size)

    def trainfn(tran: dict[str, Any], worker: int) -> None:
        if len(replay) < batch_steps:
            return
        for _ in range(should_train(step)):
            batch = next(stream_train)
            carry_train[0], outs, mets = agent.train(carry_train[0], batch)
            train_fps.step(batch_steps)
            if "replay" in outs:
                replay.update(outs["replay"])
            train_agg.add(mets, prefix="train")

    driver.on_step(trainfn)

    cp = elements.Checkpoint(logdir / "ckpt")
    cp.step = step
    cp.agent = agent
    cp.replay = replay
    cp.load_or_save()

    pending_milestones = sorted({int(s) for s in curve_steps if int(s) >= 0})
    curve: list[dict[str, Any]] = []
    best_eval: dict[str, Any] | None = None
    best_key = (-1.0, float("-inf"))
    eval_overhead_s = 0.0
    host_logdir = Path(str(config.logdir))

    def _maybe_save_best(stats: dict[str, float], cur_step: int) -> None:
        nonlocal best_eval, best_key
        cand_key = (float(stats.get("solved_rate", 0.0)), float(stats["mean_return"]))
        if cand_key < best_key:
            return
        import pickle
        import shutil

        best_key = cand_key
        best_eval = {"step": int(cur_step), **stats}
        best_dir = host_logdir / "best_ckpt"
        best_dir.mkdir(parents=True, exist_ok=True)
        (best_dir / "agent.pkl").write_bytes(pickle.dumps(agent.save()))
        for extra_name in ("policy_meta.json", "config.yaml"):
            src = host_logdir / extra_name
            if src.is_file():
                shutil.copy2(src, best_dir / extra_name)

    t0 = time.perf_counter()
    if eval_fn is not None and pending_milestones and pending_milestones[0] == 0:
        pending_milestones.pop(0)
        te0 = time.perf_counter()
        stats0 = eval_fn(agent)
        _maybe_save_best(stats0, 0)
        eval_overhead_s += time.perf_counter() - te0
        curve.append({"steps": 0, "wall_time_s": 0.0, **stats0})

    def policy(*a, **kw):
        return agent.policy(*a, mode="train", **kw)

    driver.reset(agent.init_policy)
    while int(step) < int(config.run.steps):
        driver(policy, steps=driver.length)

        if should_report(step) and len(replay) >= batch_steps:
            agg = elements.Agg()
            for _ in range(config.consec_report * config.run.report_batches):
                carry_report, mets = agent.report(carry_report, next(stream_report))
                agg.add(mets)
            logger.add(agg.result(), prefix="report")

        if should_log(step):
            logger.add(train_agg.result())
            logger.add(epstats.result(), prefix="epstats")
            logger.add(replay.stats(), prefix="replay")
            logger.add(
                {"fps/policy": policy_fps.result(), "fps/train": train_fps.result()}
            )
            logger.write()

        if should_save(step):
            cp.save()

        while (
            eval_fn is not None
            and pending_milestones
            and int(step) >= pending_milestones[0]
        ):
            target_s = pending_milestones.pop(0)
            elapsed_train = time.perf_counter() - t0 - eval_overhead_s
            te0 = time.perf_counter()
            stats = eval_fn(agent)
            _maybe_save_best(stats, int(step))
            eval_overhead_s += time.perf_counter() - te0
            curve.append(
                {
                    "steps": int(target_s),
                    "actual_steps": int(step),
                    "wall_time_s": elapsed_train,
                    **stats,
                }
            )

    wall_time = time.perf_counter() - t0 - eval_overhead_s
    cp.save()
    driver.close()
    logger.close()
    return {
        "steps": int(step),
        "wall_time_s": wall_time,
        "logdir": str(config.logdir),
        "curve": curve,
        "best_eval": best_eval,
    }


def train_mjx(
    env_name: str = "MjxElbowPoseRandom-v0",
    impl: str = "jax",
    num_envs: int = 64,
    steps: int = 500_000,
    train_ratio: float = 64.0,
    size: str = "size12m",
    platform: str = "cuda",
    seed: int = 0,
    logdir: str = "",
    outputs: Sequence[str] = ("jsonl",),
    debug: bool = False,
) -> dict[str, Any]:
    """Train DreamerV3 on a batched MyoSuite MJX / Warp environment."""
    platform = resolve_jax_platform(platform)
    resolved_logdir = logdir or str(
        Path("dreamerv3_mjx_logs")
        / f"{env_name}_{impl}_{num_envs}_seed{seed}_{int(time.time())}"
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
    )
    elements.Path(config.logdir).mkdir()
    config.save(elements.Path(config.logdir) / "config.yaml")

    mjx_env = load_myosuite_mjx_env(env_name, num_envs=num_envs, impl=impl)
    driver = MJXDriver(mjx_env, num_envs=num_envs, seed=seed)
    write_policy_metadata(
        config.logdir,
        config,
        driver.obs_space,
        driver.act_space,
        env_id=env_name,
    )
    agent, make_replay, make_stream, make_logger = make_agent_and_factories(
        config, driver.obs_space, driver.act_space
    )
    summary = run_mjx_training_loop(
        driver, agent, make_replay, make_stream, make_logger, config
    )
    summary.update(
        {"env_name": env_name, "impl": impl, "num_envs": num_envs, "seed": seed}
    )
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """CLI entrypoint for ``dreamerv3-train-mjx``."""
    parser = argparse.ArgumentParser(
        description="Train DreamerV3 on batched GPU MyoSuite MJX / Warp environments."
    )
    parser.add_argument("--env-name", type=str, default="MjxElbowPoseRandom-v0")
    parser.add_argument("--impl", type=str, default="jax", choices=("jax", "warp"))
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--train-ratio", type=float, default=64.0)
    parser.add_argument(
        "--size",
        type=str,
        default="size12m",
        choices=("size1m", "size12m", "size25m", "size50m"),
    )
    parser.add_argument(
        "--platform", type=str, default="cuda", choices=("cuda", "cpu", "tpu")
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    result = train_mjx(
        env_name=args.env_name,
        impl=args.impl,
        num_envs=args.num_envs,
        steps=args.steps,
        train_ratio=args.train_ratio,
        size=args.size,
        platform=args.platform,
        seed=args.seed,
        logdir=args.logdir,
        debug=args.debug,
    )
    print(
        f"Completed {result['steps']} steps on {args.env_name} ({args.impl}, "
        f"{args.num_envs} envs) in {result['wall_time_s']:.2f}s."
    )
    return result


if __name__ == "__main__":
    main()
