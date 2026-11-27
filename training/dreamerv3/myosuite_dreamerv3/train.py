# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Train and evaluate DreamerV3 on MyoSuite CPU (Gymnasium) environments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from functools import partial as bind
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

import elements
import embodied
import numpy as np
import ruamel.yaml as yaml

from myosuite_dreamerv3.compat import (
    apply_dreamer_jax_compat,
    import_dreamerv3,
    import_myosuite,
    resolve_jax_platform,
)
from myosuite_dreamerv3.env import DreamerEnv, wrap_dreamer_env
from myosuite_dreamerv3.export import (
    evaluate_dreamer_torch_policy,
    export_dreamerv3_onnx,
    export_dreamerv3_torchscript,
    load_dreamerv3_torch_policy,
)
from myosuite_dreamerv3.hf import resolve_dreamerv3_checkpoint


def build_dreamer_config(
    size: str = "size12m",
    platform: str = "cuda",
    seed: int = 0,
    steps: int = 500_000,
    envs: int = 8,
    eval_envs: int = 2,
    train_ratio: float = 512.0,
    batch_size: int | None = None,
    batch_length: int | None = None,
    logdir: str | Path = "",
    from_checkpoint: str = "",
    outputs: Sequence[str] = ("jsonl",),
    debug: bool = False,
) -> elements.Config:
    """Build an ``elements.Config`` from DreamerV3's ``configs.yaml``."""
    apply_dreamer_jax_compat()
    resolved_platform = (
        "cpu" if debug and platform == "cuda" else resolve_jax_platform(platform)
    )
    _, cfg_path = import_dreamerv3()
    configs = yaml.YAML(typ="safe").load(cfg_path.read_text(encoding="utf-8"))
    config = elements.Config(configs["defaults"])
    config = config.update(configs[size])
    if debug:
        config = config.update(configs["debug"])

    resolved_logdir = str(logdir or Path("dreamerv3_logs") / f"run_{int(time.time())}")
    overrides: dict[str, Any] = {
        "seed": int(seed),
        "logdir": resolved_logdir,
        "logger.outputs": list(outputs),
        "jax.platform": resolved_platform,
        "jax.prealloc": False if resolved_platform == "cpu" or debug else True,
        "run.steps": int(steps),
        "run.envs": int(envs),
        "run.eval_envs": int(eval_envs),
        "run.train_ratio": float(train_ratio),
    }
    if from_checkpoint:
        overrides["run.from_checkpoint"] = str(from_checkpoint)
        if "from_checkpoint_regex" not in config.run:
            config = config.update(
                {"run": {**dict(config.run), "from_checkpoint_regex": ".*"}}
            )
    if batch_size is not None:
        overrides["batch_size"] = int(batch_size)
    if batch_length is not None:
        overrides["batch_length"] = int(batch_length)
    return config.update(overrides)


def write_policy_metadata(
    logdir: str | Path,
    config: elements.Config,
    obs_space: dict[str, elements.Space],
    act_space: dict[str, elements.Space],
    *,
    env_id: str = "",
) -> Path | None:
    """Write ``policy_meta.json`` for standalone PyTorch / ONNX reconstruction."""
    if "proprio" not in obs_space or "image" in obs_space:
        return None
    rssm_cfg = config.agent.dyn.rssm
    enc_cfg = config.agent.enc.simple
    pol_cfg = config.agent.policy
    meta = {
        "env_id": env_id,
        "obs_dim": int(np.prod(obs_space["proprio"].shape)),
        "act_dim": int(np.prod(act_space["action"].shape)),
        "deter": int(rssm_cfg.deter),
        "stoch": int(rssm_cfg.stoch),
        "classes": int(rssm_cfg.classes),
        "blocks": int(rssm_cfg.blocks),
        "symlog": bool(enc_cfg.symlog),
        "enc_act": str(enc_cfg.act),
        "dyn_act": str(rssm_cfg.act),
        "pol_act": str(pol_cfg.act),
        "norm_eps": 1e-4,
    }
    out = Path(str(logdir)) / "policy_meta.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def _resolve_agent_ckpt_file(ckpt_root: Path) -> Path:
    if ckpt_root.is_file():
        return ckpt_root.parent if ckpt_root.name == "agent.pkl" else ckpt_root
    if (ckpt_root / "agent.pkl").is_file():
        return ckpt_root
    if (ckpt_root / "latest").is_file():
        latest = (ckpt_root / "latest").read_text(encoding="utf-8").strip()
        return ckpt_root / latest
    if (ckpt_root / "ckpt" / "latest").is_file():
        latest = (ckpt_root / "ckpt" / "latest").read_text(encoding="utf-8").strip()
        return ckpt_root / "ckpt" / latest
    raise FileNotFoundError(
        f"Could not find DreamerV3 checkpoint folder under {ckpt_root}"
    )


def make_agent_and_factories(
    config: elements.Config,
    obs_space: dict[str, elements.Space],
    act_space: dict[str, elements.Space],
) -> tuple[Any, Any, Any, Any]:
    """Instantiate DreamerV3 ``Agent`` plus replay, stream, and logger factories."""
    apply_dreamer_jax_compat()
    import jax
    from dreamerv3.agent import Agent

    if config.jax.platform:
        try:
            jax.config.update("jax_platforms", config.jax.platform)
        except Exception:
            pass

    def notlog(k):
        return not k.startswith("log/")

    agent_obs = {k: v for k, v in obs_space.items() if notlog(k)}
    agent_act = {k: v for k, v in act_space.items() if k != "reset"}
    agent = Agent(
        agent_obs,
        agent_act,
        elements.Config(
            **config.agent,
            logdir=config.logdir,
            seed=config.seed,
            jax=config.jax,
            batch_size=config.batch_size,
            batch_length=config.batch_length,
            replay_context=config.replay_context,
            report_length=config.report_length,
            replica=config.replica,
            replicas=config.replicas,
        ),
    )

    def make_replay(folder: str = "replay", mode: str = "train") -> Any:
        batlen = config.batch_length if mode == "train" else config.report_length
        consec = config.consec_train if mode == "train" else config.consec_report
        capacity = int(config.replay.size // (1 if mode == "train" else 10))
        length = consec * batlen + config.replay_context
        return embodied.replay.Replay(
            length=length,
            capacity=capacity,
            directory=elements.Path(config.logdir) / folder,
            online=config.replay.online,
            chunksize=config.replay.chunksize,
        )

    def make_stream(replay: Any, mode: str) -> Any:
        fn = bind(replay.sample, config.batch_size, mode)
        stream = embodied.streams.Stateless(fn)
        stream = embodied.streams.Consec(
            stream,
            length=config.batch_length if mode == "train" else config.report_length,
            consec=config.consec_train if mode == "train" else config.consec_report,
            prefix=config.replay_context,
            strict=(mode == "train"),
            contiguous=True,
        )
        return stream

    def make_logger() -> Any:
        step = elements.Counter()
        logdir = config.logdir
        outputs = [elements.logger.TerminalOutput(config.logger.filter, "Agent")]
        for output in config.logger.outputs:
            if output == "jsonl":
                outputs.append(elements.logger.JSONLOutput(logdir, "metrics.jsonl"))
                outputs.append(
                    elements.logger.JSONLOutput(logdir, "scores.jsonl", "episode/score")
                )
            elif output == "scope":
                outputs.append(elements.logger.ScopeOutput(elements.Path(logdir)))
            elif output == "tensorboard":
                outputs.append(elements.logger.TensorBoardOutput(logdir))
            elif output == "wandb":
                outputs.append(
                    elements.logger.WandBOutput(
                        f"{Path(logdir).name}", config=dict(config)
                    )
                )
        return elements.Logger(step, outputs, multiplier=1)

    return agent, make_replay, make_stream, make_logger


def evaluate_dreamer_agent(
    agent: Any,
    env_id: str,
    episodes: int = 10,
    seed: int = 0,
    vision: bool = False,
    features: Sequence[Any] = (),
    deterministic: bool = True,
    meta: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Evaluate a DreamerV3 ``agent`` over ``episodes`` rollouts."""
    if deterministic and not vision:
        try:
            saved = agent.save()
            rssm_cfg = getattr(getattr(agent.config, "dyn", None), "rssm", None)
            enc_cfg = getattr(getattr(agent.config, "enc", None), "simple", None)
            pol_cfg = getattr(agent.config, "policy", None)
            inferred_meta = dict(meta or {})
            if rssm_cfg is not None:
                inferred_meta.setdefault("stoch", int(rssm_cfg.stoch))
                inferred_meta.setdefault("classes", int(rssm_cfg.classes))
                inferred_meta.setdefault("blocks", int(rssm_cfg.blocks))
                inferred_meta.setdefault("dyn_act", str(rssm_cfg.act))
            if enc_cfg is not None:
                inferred_meta.setdefault("enc_act", str(enc_cfg.act))
                inferred_meta.setdefault("symlog", bool(enc_cfg.symlog))
            if pol_cfg is not None:
                inferred_meta.setdefault("pol_act", str(pol_cfg.act))
            pt_pol = load_dreamerv3_torch_policy(saved, meta=inferred_meta)
            return evaluate_dreamer_torch_policy(
                pt_pol,
                env_id=env_id,
                episodes=episodes,
                seed=seed,
                features=features,
            )
        except Exception:
            pass
    returns: list[float] = []
    solved_flags: list[float] = []
    lengths: list[int] = []

    def notlog(k):
        return not k.startswith("log/")

    for ep in range(episodes):
        raw_env = DreamerEnv(
            env_id,
            proprio=True,
            image=vision,
            seed=int(seed + ep),
            features=features,
        )
        env = wrap_dreamer_env(raw_env)
        try:
            carry = agent.init_policy(1)
            obs = env.step(
                {
                    "action": np.zeros(env.act_space["action"].shape, np.float32),
                    "reset": np.bool_(True),
                }
            )
            ep_return = 0.0
            ep_solved = False
            ep_len = 0
            while not obs["is_last"]:
                batched_obs = {
                    k: np.expand_dims(np.asarray(v), 0)
                    for k, v in obs.items()
                    if notlog(k)
                }
                carry, acts, _ = agent.policy(carry, batched_obs, mode="eval")
                act = np.asarray(acts["action"][0], np.float32)
                obs = env.step({"action": act, "reset": np.bool_(False)})
                ep_return += float(obs["reward"])
                ep_solved = ep_solved or (float(obs["log/solved"]) > 0.5)
                ep_len += 1
            returns.append(ep_return)
            solved_flags.append(1.0 if ep_solved else 0.0)
            lengths.append(ep_len)
        finally:
            env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "solved_rate": float(np.mean(solved_flags)),
        "mean_length": float(np.mean(lengths)),
    }


def train_cpu(
    env_id: str = "myoElbowPose1D6MRandom-v0",
    steps: int = 500_000,
    envs: int = 8,
    eval_envs: int = 2,
    train_ratio: float = 512.0,
    size: str = "size12m",
    vision: bool = False,
    platform: str = "cuda",
    seed: int = 0,
    logdir: str = "",
    from_checkpoint: str = "",
    checkpoint_revision: str | None = None,
    eval_only: bool = False,
    export_onnx: str = "",
    export_torchscript: str = "",
    outputs: Sequence[str] = ("jsonl",),
    script: str = "train",
    features: Sequence[Any] = (),
    debug: bool = False,
    eval_episodes: int = 0,
    curve_steps: Sequence[int] = (),
) -> dict[str, Any]:
    """Train or evaluate DreamerV3 on a MyoSuite CPU Gymnasium environment."""
    import_myosuite()
    resolved_ckpt = ""
    if from_checkpoint:
        ckpt_dir = _resolve_agent_ckpt_file(
            resolve_dreamerv3_checkpoint(from_checkpoint, revision=checkpoint_revision)
        )
        resolved_ckpt = str(ckpt_dir)

    resolved_logdir = logdir or str(
        Path("dreamerv3_logs") / f"{env_id}_seed{seed}_{int(time.time())}"
    )
    config = build_dreamer_config(
        size=size,
        platform=platform,
        seed=seed,
        steps=steps,
        envs=envs,
        eval_envs=eval_envs,
        train_ratio=train_ratio,
        logdir=resolved_logdir,
        from_checkpoint=resolved_ckpt,
        outputs=outputs,
        debug=debug,
    )
    elements.Path(config.logdir).mkdir()
    config.save(elements.Path(config.logdir) / "config.yaml")

    probe = wrap_dreamer_env(
        DreamerEnv(env_id, proprio=True, image=vision, seed=seed, features=features)
    )
    obs_space = dict(probe.obs_space)
    act_space = dict(probe.act_space)
    probe.close()

    meta_path = write_policy_metadata(
        config.logdir, config, obs_space, act_space, env_id=env_id
    )
    agent, make_replay, make_stream, make_logger = make_agent_and_factories(
        config, obs_space, act_space
    )
    if resolved_ckpt:
        elements.checkpoint.load(
            resolved_ckpt,
            dict(agent=bind(agent.load, regex=r".*")),
        )

    if eval_only:
        eval_stats = evaluate_dreamer_agent(
            agent,
            env_id=env_id,
            episodes=max(1, eval_episodes),
            seed=seed + 10_000,
            vision=vision,
            features=features,
        )
        return {
            "env_id": env_id,
            "seed": seed,
            "steps": 0,
            "wall_time_s": 0.0,
            "logdir": config.logdir,
            "eval": eval_stats,
        }

    def make_env_fn(index: int = 0) -> embodied.Env:
        seed_seq = np.random.SeedSequence([int(seed), int(index)])
        env_seed = int(seed_seq.generate_state(1)[0]) % (2**32 - 1)
        return wrap_dreamer_env(
            DreamerEnv(
                env_id,
                proprio=True,
                image=vision,
                seed=env_seed,
                features=features,
            )
        )

    curve: list[dict[str, Any]] = []
    if curve_steps:
        from myosuite_dreamerv3.train_mjx import run_mjx_training_loop

        driver = embodied.Driver(
            [bind(make_env_fn, i) for i in range(int(envs))], parallel=False
        )
        eval_cb = (
            (
                lambda ag: evaluate_dreamer_agent(
                    ag,
                    env_id=env_id,
                    episodes=max(1, eval_episodes),
                    seed=seed + 10_000,
                    vision=vision,
                    features=features,
                )
            )
            if eval_episodes > 0
            else None
        )
        loop_res = run_mjx_training_loop(
            driver,
            agent,
            make_replay,
            make_stream,
            make_logger,
            config,
            eval_fn=eval_cb,
            curve_steps=curve_steps,
        )
        wall_time = loop_res["wall_time_s"]
        actual_steps = loop_res["steps"]
        curve = loop_res["curve"]
    else:
        holder = [agent]
        args = elements.Config(
            **config.run,
            replica=config.replica,
            replicas=config.replicas,
            logdir=config.logdir,
            batch_size=config.batch_size,
            batch_length=config.batch_length,
            report_length=config.report_length,
            consec_train=config.consec_train,
            consec_report=config.consec_report,
            replay_context=config.replay_context,
        )
        t0 = time.perf_counter()
        if script == "train_eval":
            embodied.run.train_eval(
                lambda: holder[0],
                bind(make_replay, "replay", "train"),
                bind(make_replay, "eval_replay", "eval"),
                make_env_fn,
                make_env_fn,
                make_stream,
                make_logger,
                args,
            )
        else:
            embodied.run.train(
                lambda: holder[0],
                bind(make_replay, "replay", "train"),
                make_env_fn,
                make_stream,
                make_logger,
                args,
            )
        wall_time = time.perf_counter() - t0
        actual_steps = steps
        latest_file = Path(str(config.logdir)) / "ckpt" / "latest"
        if latest_file.is_file():
            import pickle

            latest_dir = (
                latest_file.parent / latest_file.read_text(encoding="utf-8").strip()
            )
            if latest_dir.is_dir():
                (latest_dir / "agent.pkl").write_bytes(pickle.dumps(agent.save()))

    summary: dict[str, Any] = {
        "env_id": env_id,
        "seed": seed,
        "steps": actual_steps,
        "wall_time_s": wall_time,
        "logdir": config.logdir,
        "curve": curve,
    }
    if eval_episodes > 0:
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
                vision=vision,
                features=features,
            )
    if not vision and meta_path is not None:
        best_dir = Path(str(config.logdir)) / "best_ckpt"
        export_src = best_dir if (best_dir / "agent.pkl").is_file() else config.logdir
        if export_torchscript:
            export_dreamerv3_torchscript(export_src, export_torchscript)
            summary["torchscript"] = str(Path(export_torchscript).resolve())
        if export_onnx:
            export_dreamerv3_onnx(
                export_src, export_onnx, env_id=env_id, features=features
            )
            summary["onnx"] = str(Path(export_onnx).resolve())
    return summary


def parse_args(
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, tuple[Any, ...]]:
    """Parse CLI arguments and MyoSuite ``--feature`` flags."""
    from myosuite.utils.feature_cli import parse_feature_args

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    features, rest = parse_feature_args(raw_argv)

    parser = argparse.ArgumentParser(
        description="Train and evaluate DreamerV3 on MyoSuite CPU environments."
    )
    parser.add_argument("--env-id", type=str, default="myoElbowPose1D6MRandom-v0")
    parser.add_argument(
        "--size",
        type=str,
        default="size12m",
        choices=("size1m", "size12m", "size25m", "size50m"),
    )
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--eval-envs", type=int, default=2)
    parser.add_argument("--train-ratio", type=float, default=512.0)
    parser.add_argument("--vision", action="store_true")
    parser.add_argument(
        "--platform", type=str, default="cuda", choices=("cuda", "cpu", "tpu")
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="")
    parser.add_argument("--from-checkpoint", type=str, default="")
    parser.add_argument("--checkpoint-revision", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--export-onnx", type=str, default="")
    parser.add_argument("--export-torchscript", type=str, default="")
    parser.add_argument(
        "--script", type=str, default="train", choices=("train", "train_eval")
    )
    parser.add_argument(
        "--outputs",
        type=str,
        default="jsonl,scope" if importlib.util.find_spec("scope") else "jsonl",
    )
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(rest), features


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """CLI entrypoint for ``dreamerv3-train``."""
    args, features = parse_args(argv)
    outputs = tuple(x.strip() for x in args.outputs.split(",") if x.strip())
    return train_cpu(
        env_id=args.env_id,
        steps=args.steps,
        envs=args.envs,
        eval_envs=args.eval_envs,
        train_ratio=args.train_ratio,
        size=args.size,
        vision=args.vision,
        platform=args.platform,
        seed=args.seed,
        logdir=args.logdir,
        from_checkpoint=args.from_checkpoint,
        checkpoint_revision=args.checkpoint_revision,
        eval_only=args.eval_only,
        export_onnx=args.export_onnx,
        export_torchscript=args.export_torchscript,
        outputs=outputs,
        script=args.script,
        features=features,
        debug=args.debug,
        eval_episodes=args.eval_episodes,
    )


if __name__ == "__main__":
    main()
