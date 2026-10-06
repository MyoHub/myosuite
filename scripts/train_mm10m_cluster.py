#!/usr/bin/env python3
"""Train a general full-body MuscleMimic policy with mjlab at the scale of ``mm-10m-2``.

``amathislab/mm-10m-2`` (tutorial 5.1) was trained for 2.048B steps on 8192 envs over the 972 KIT motions
of ``KIT_KINESIS_TRAINING_MOTIONS`` and validated on the 108 clips of ``KIT_KINESIS_TESTING_MOTIONS``.
This script runs the same recipe on MyoSuite's mjlab Mimic task (``myoMimicFullbody-v0`` with a clip bank).
It is an approximation, not a bit-exact re-run (see "Differences" below).

Setup (once, on a node with internet access)::

    pip install "myosuite[mjlab,musclemimic] @ git+https://github.com/MyoHub/myosuite.git@ms3"
    export HF_TOKEN=...            # the clip dataset is gated
    python scripts/train_mm10m_cluster.py --download-only --data-dir $SCRATCH/mm_clips

Train (single GPU, or ``--nproc_per_node=<gpus>``; ``--num-envs-total`` is split over the GPUs)::

    torchrun --standalone --nproc_per_node=4 scripts/train_mm10m_cluster.py \
        --data-dir $SCRATCH/mm_clips --out-dir $SCRATCH/mm_runs --run-name mm10m_mjlab

Resume after a time limit (re-submit the same command, it continues from the newest checkpoint)::

    ... --resume --time-limit-min 1400     # exits with code 3 when it stopped on the time limit

Check the install first with ``--smoke`` (8 clips, 64 envs, 3 iterations), and evaluate a checkpoint on the
held-out clips with ``--eval <model_N.pt>`` (single process). ``--print-slurm`` prints a job script.

Differences to the original recipe (what is kept): total steps, global batch (envs x 20 steps), 1 epoch,
minibatches of 1280, lr 4e-4 annealed linearly to 10%, gamma 0.99, GAE 0.95, clip 0.2, value coef 0.5,
entropy 0, init std 3.0, 5 x 1024 SiLU actor/critic, observation normalization, clip bank = the KIT training split.
What differs: the observation (mjlab clip layout, 1152-d, not their 2418-d), no residual blocks / LayerNorm /
Muon optimizer (rsl_rl: plain MLP + Adam), the reward and termination of the mjlab Mimic task.
Memory: 8192 envs (one GPU, 5 x 1024 network, 8 clips) used about 20 GB on a 32 GB RTX 5090; the script prints the
peak torch memory after the first iteration (Warp buffers are not included, check ``nvidia-smi``). Tune
``--num-envs-total`` / ``--nconmax`` / ``--njmax`` if needed. Multi-GPU (NCCL via torchrun) is untested.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MOTIONS = HERE / "mm10m_motions"
HF_REPO = "amathislab/musclemimic-retargeted"
TASK = "myoMimicFullbody-v0"
SLURM = """#!/bin/bash
#SBATCH --job-name=mm10m
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --requeue
export MUJOCO_GL=egl OMP_NUM_THREADS=4
srun torchrun --standalone --nproc_per_node=4 scripts/train_mm10m_cluster.py \\
    --data-dir $SCRATCH/mm_clips --out-dir $SCRATCH/mm_runs --run-name mm10m_mjlab --resume --time-limit-min 1380
# exit code 3 = stopped on the time limit: re-submit (or `scontrol requeue`) to continue
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("mm_clips"), help="Clip folder (HF layout).")
    p.add_argument("--out-dir", type=Path, default=Path("mm_runs"))
    p.add_argument("--run-name", default="mm10m_mjlab")
    p.add_argument("--split", choices=["training", "testing"], default="training")
    p.add_argument("--max-clips", type=int, default=None, help="Use only the first N clips (debugging).")
    p.add_argument("--download-only", action="store_true", help="Download the clips and exit.")
    p.add_argument("--num-envs-total", type=int, default=8192)
    p.add_argument("--total-steps", type=float, default=2.048e9)
    p.add_argument("--steps-per-env", type=int, default=20)
    p.add_argument("--minibatch-size", type=int, default=1280)
    p.add_argument("--hidden", type=int, nargs="+", default=[1024] * 5)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--init-std", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nconmax", type=int, default=None, help="Override the contact buffer per world.")
    p.add_argument("--njmax", type=int, default=None, help="Override the constraint buffer per world.")
    p.add_argument("--save-interval", type=int, default=100, help="Iterations between checkpoints.")
    p.add_argument("--keep-every", type=int, default=5000, help="Keep every N-th checkpoint besides the last 3.")
    p.add_argument("--logger", choices=["tensorboard", "wandb"], default="tensorboard")
    p.add_argument("--wandb-project", default="mm10m", help="W&B project (use WANDB_MODE=offline on nodes without internet).")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--time-limit-min", type=float, default=None)
    p.add_argument("--smoke", action="store_true", help="8 clips, 64 envs, 3 iterations.")
    p.add_argument("--eval", type=Path, default=None, help="Evaluate this checkpoint on the testing clips.")
    p.add_argument("--print-slurm", action="store_true")
    return p.parse_args()


def clip_keys(split: str, limit: int | None) -> list[str]:
    keys = [k.strip() for k in (MOTIONS / f"kit_kinesis_{split}.txt").read_text().splitlines() if k.strip()]
    return keys[:limit] if limit else keys


def download(keys: list[str], data_dir: Path) -> list[Path]:
    """Return the local npz path of every key, downloading what is missing."""
    paths = [data_dir / "MyoFullBody" / "gmr" / f"{k}.npz" for k in keys]
    missing = [k for k, path in zip(keys, paths) if not path.is_file()]
    if missing:
        from huggingface_hub import snapshot_download

        print(f"[data] downloading {len(missing)} clips to {data_dir}", flush=True)
        snapshot_download(
            repo_id=HF_REPO, repo_type="dataset", local_dir=str(data_dir),
            allow_patterns=[f"MyoFullBody/gmr/{k}.npz" for k in missing],
        )
    return paths


def build_cfg_dict(args, per_gpu_envs: int, world: int, iters: int):
    """rsl_rl runner config (as dict) with the mm-10m-2 hyper-parameters."""
    from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

    batch = per_gpu_envs * args.steps_per_env
    dist = {"class_name": "GaussianDistribution", "init_std": args.init_std, "std_type": "scalar"}
    cfg = RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=tuple(args.hidden), activation="swish", obs_normalization=True,
                            distribution_cfg=dist),  # rsl_rl calls SiLU "swish"
        critic=RslRlModelCfg(hidden_dims=tuple(args.hidden), activation="swish", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            num_learning_epochs=1, num_mini_batches=max(1, round(batch / args.minibatch_size)),
            learning_rate=args.lr, schedule="fixed",  # annealed by the training loop
            gamma=0.99, lam=0.95, clip_param=0.2, value_loss_coef=0.5, entropy_coef=0.0,
            max_grad_norm=1.0, use_clipped_value_loss=True, normalize_advantage_per_mini_batch=False,
        ),
        num_steps_per_env=args.steps_per_env, max_iterations=iters, seed=args.seed,
        save_interval=args.save_interval, experiment_name=args.run_name, logger=args.logger,
        wandb_project=args.wandb_project, run_name=args.run_name,
    )
    return cfg, dataclasses.asdict(cfg)


def make_env(args, paths: list[Path], num_envs: int, device: str, seed: int, rl_cfg):
    """Register the clip-bank task and build the (wrapped) mjlab env."""
    import mjlab.tasks.registry as registry
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    from myosuite.core.trajectory_io import load_motion_clip
    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import register_mimic_mjlab_tasks_with_clip

    t0 = time.time()
    clips = []
    for i, path in enumerate(paths):
        clips.append(load_motion_clip(path, expected_nq=89, expected_nv=88))
        if i % 100 == 0:
            print(f"[data] loaded {i + 1}/{len(paths)} clips ({time.time() - t0:.0f} s)", flush=True)
    register_mimic_mjlab_tasks_with_clip(
        register_mjlab_task=registry.register_mjlab_task, rl_cfg_fn=lambda: rl_cfg, clip=tuple(clips),
        use_lookahead=True,
    )
    env_cfg = registry.load_env_cfg(TASK)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    if args.nconmax:
        env_cfg.sim.nconmax = args.nconmax
    if args.njmax:
        env_cfg.sim.njmax = args.njmax
    return RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))


def checkpoints(log_dir: Path) -> list[tuple[int, Path]]:
    found = [(int(m.group(1)), p) for p in log_dir.glob("model_*.pt") if (m := re.search(r"model_(\d+)\.pt$", p.name))]
    return sorted(found)


def prune(log_dir: Path, keep_every: int) -> None:
    for it, path in checkpoints(log_dir)[:-3]:
        if it % keep_every != 0:
            path.unlink(missing_ok=True)


def evaluate(args, device: str) -> None:
    import numpy as np
    import torch

    keys = clip_keys("testing", args.max_clips)
    paths = download(keys, args.data_dir)
    rl_cfg, cfg_dict = build_cfg_dict(args, 100, 1, 1)
    env = make_env(args, paths, 100, device, args.seed, rl_cfg)
    from mjlab.rl import MjlabOnPolicyRunner

    runner = MjlabOnPolicyRunner(env=env, train_cfg=cfg_dict, log_dir=None, device=device)
    runner.load(str(args.eval), map_location=device)
    policy = runner.get_inference_policy(device=device)
    obs = env.get_observations()
    n = env.num_envs
    ended = np.full(n, -1)
    early = np.zeros(n, bool)
    for t in range(1500):
        with torch.no_grad():
            obs, _, dones, extras = env.step(policy(obs))
        d = dones.cpu().numpy().astype(bool)
        timeout = extras.get("time_outs", torch.zeros_like(dones)).cpu().numpy().astype(bool)
        new = (ended < 0) & d
        ended[new], early[new] = t + 1, ~timeout[new]
        if (ended >= 0).all():
            break
    done = ended >= 0
    print(f"[eval] {done.sum()}/{n} episodes ended | reached the clip end (no early termination): "
          f"{int((done & ~early).sum())} | terminated early: {int(early.sum())} | mean length {ended[done].mean():.1f}")


def main() -> int:
    args = parse_args()
    if args.print_slurm:
        print(SLURM)
        return 0
    if args.smoke:
        args.max_clips, args.num_envs_total, args.total_steps, args.save_interval = 8, 64, 3 * 64 * args.steps_per_env, 3
    if args.download_only:
        for split in ("training", "testing"):
            download(clip_keys(split, args.max_clips), args.data_dir)
        return 0

    import torch

    rank, local, world = (int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("LOCAL_RANK", 0), ("WORLD_SIZE", 1)))
    device = f"cuda:{local}"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local)
    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.cuda.set_device(local)
    if args.eval:
        evaluate(args, device)
        return 0

    per_gpu = max(1, args.num_envs_total // world)
    iters = math.ceil(args.total_steps / (per_gpu * world * args.steps_per_env))
    log_dir = args.out_dir / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    keys = clip_keys(args.split, args.max_clips)
    paths = download(keys, args.data_dir)
    if rank == 0:
        print(f"[run] {len(keys)} clips, {world} GPU(s) x {per_gpu} envs, {iters} iterations of "
              f"{per_gpu * world * args.steps_per_env} steps = {iters * per_gpu * world * args.steps_per_env:.3e}", flush=True)

    from mjlab.rl import MjlabOnPolicyRunner
    from mjlab.utils.torch import configure_torch_backends

    from myosuite.envs.myo.backends.mjlab.rsl_rl_logger_episode_patch import install_episode_reward_logging_patch

    configure_torch_backends()
    rl_cfg, cfg_dict = build_cfg_dict(args, per_gpu, world, iters)
    env = make_env(args, paths, per_gpu, device, args.seed + rank, rl_cfg)
    install_episode_reward_logging_patch()
    runner = MjlabOnPolicyRunner(env=env, train_cfg=cfg_dict, log_dir=str(log_dir), device=device)
    # rsl_rl's counter is the index of the last finished iteration (also in a loaded checkpoint),
    # so the next iteration to run is that + 1.
    done_it = 0
    if args.resume and checkpoints(log_dir):
        _, path = checkpoints(log_dir)[-1]
        runner.load(str(path), map_location=device)
        done_it = runner.current_learning_iteration + 1
        print(f"[run] resumed from {path.name}, continuing at iteration {done_it}", flush=True)
    runner.current_learning_iteration = done_it

    # One learn() call (one TensorBoard/W&B run); the lr schedule, checkpoint pruning and the
    # time limit run in a hook around the PPO update, which is called once per iteration.
    state = {"it": done_it, "start": time.time()}
    update = runner.alg.update

    class TimeLimit(Exception):
        pass

    def update_hook(*a, **kw):
        it = state["it"]
        lr = args.lr * (1.0 - (1.0 - args.min_lr_ratio) * it / iters)
        runner.alg.learning_rate = lr
        for group in runner.alg.optimizer.param_groups:
            group["lr"] = lr
        out = update(*a, **kw)
        state["it"] = it + 1
        if rank == 0:
            if it == done_it:
                print(f"[run] peak GPU memory after the first iteration: "
                      f"{torch.cuda.max_memory_allocated(local) / 2**30:.1f} GiB", flush=True)
            if it % args.save_interval == 0:
                prune(log_dir, args.keep_every)
        if args.time_limit_min and (time.time() - state["start"]) / 60 > args.time_limit_min:
            runner.current_learning_iteration = it
            raise TimeLimit
        return out

    runner.alg.update = update_hook
    try:
        runner.learn(num_learning_iterations=iters - done_it, init_at_random_ep_len=done_it == 0)
    except TimeLimit:
        runner.save(str(log_dir / f"model_{runner.current_learning_iteration}.pt"))
        if rank == 0:
            print(f"[run] time limit reached after iteration {runner.current_learning_iteration}/{iters}", flush=True)
        return 3
    if rank == 0:
        print("[run] training complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
