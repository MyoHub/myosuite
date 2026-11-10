# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# training/supermimic/ for the notices of the code it includes.
"""SuperMimic runner configuration (paper Table A.5 / supermimic_vibed defaults).

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at commit
8a6244b, ``src/muscle_supermimic/rl/supermimic/config.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class SuperMimicRunnerCfg(RslRlBaseRunnerCfg):
    """SuperMimic runner settings. The field defaults are SuperMimic's (muscle-supermimic's first release);
    :func:`fast_runner_cfg` sets the fast recipe."""

    class_name: str = "SuperMimicRunner"
    obs_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {"policy": ("policy",), "goal": ("goal",)}
    )
    """Observation groups: ``policy`` is the decoder's proprioception history, ``goal`` the encoder's input."""
    command_name: str = "motion"
    soft_width: float = 0.05
    """Width of the sigmoid that softens the task's failure rule on imagined states, relative to each threshold
    (absolute for the full body, whose thresholds are 1 m and 1 rad)."""
    max_iterations: int = 320_000
    save_interval: int = 2000
    log_interval: int = 50
    logger: Literal["wandb", "tensorboard"] = "tensorboard"

    # data collection
    buffer_steps: int = 1000
    """Replay buffer length per env (the buffer holds buffer_steps x num_envs transitions)."""
    utd: float = 0.4
    """Updates per vectorized simulator step (2.5 sim steps per iteration)."""
    action_noise: float = 0.1
    """Exploration noise N(0, sigma^2) added to the action, i.e. the excitations, during collection, then clipped to
    [0, 1]. Noise on the logits would shrink to sigma * e(1 - e) at low excitations."""
    action_noise_dist: Literal["gaussian", "beta", "reflect"] = "gaussian"
    """Distribution of the exploration noise (networks.explore): "gaussian" adds N(0, action_noise^2) and clips to [0, 1],
    which biases near-silent muscles upwards; "beta" draws from a Beta distribution with mean the policy's excitation and
    standard deviation action_noise at 0.5, shrinking towards 0 and 1, so it stays in the box with no bias (Trial 10:
    the policy stalls, near-silent muscles are never driven up); "reflect" adds N(0, action_noise^2) and folds the draws
    back into [0, 1] at the bounds: the Gaussian's step sizes, nothing piled on 0 or 1, biased upwards near 0."""
    noise_free_frac: float = 0.0
    """Share of the training envs (the first ones) that collect with the noise-free policy, so that the replay buffer and
    imagination's start states cover the deployed policy's own states; their episode returns are logged as det_ret /
    det_len (a live deterministic training curve). 0: every env explores."""
    budget_schedules: bool = False
    """With max_hours > 0: the cosine learning-rate decay runs over the time budget (the runner's clock), ending at 3 %,
    instead of over lr_decay_iters, and the action noise anneals from action_noise to action_noise_final on the same
    cosine."""
    action_noise_final: float = 0.02
    """The action noise at the end of the time budget (budget_schedules)."""
    warmup_steps: int = 64
    """Simulator steps before any update."""

    # world model
    wm_hidden: tuple[int, ...] = (2048, 2048, 2048)
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 2048
    wm_only_iters: int = 500
    """World-model-only iterations at the start."""
    touch_loss_weight: float = 1.0
    """Weight of the world model's foot-touch loss."""

    # policy and critic
    enc_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    dec_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    latent_dim: int = 32
    critic_hidden: tuple[int, ...] = (256, 128)
    init_excitation: float = 0.1
    """Initial policy output: every muscle at this excitation (sigmoid of the decoder's output bias)."""
    horizon: int = 32
    pi_batch: int = 4096
    gamma: float = 0.99
    lam: float = 0.95
    critic_epochs: int = 5
    critic_minibatches: int = 4
    use_critic: bool = True
    remat_imagination: bool = True
    """Recompute each imagined step in the backward pass: the muscle geometry of every step would otherwise stay in
    memory."""
    critic_target_alpha: float = 0.0
    """Target critic for the policy's bootstrap and the lambda-return targets, V' <- alpha V' + (1 - alpha) V
    after each critic update (as in SHAC). 0 uses the online critic, as SuperMimic does."""
    value_clip: bool = False
    """Clamp bootstrapped values to the largest attainable return, max step reward / (1 - gamma), so the policy
    cannot exploit critic overestimation beyond it. Off in SuperMimic."""
    state_grad_clip: float = 100.0
    """Not in SuperMimic (where it is 0, off): at every imagined step, clip each trajectory's gradient with respect
    to what is carried to the next step (body state, activation, touch, last action) to this norm, measured on that
    trajectory's own objective (the loss averages pi_batch of them). It bounds the closed-loop growth of the BPTT
    gradient (observation -> policy -> activation -> world model -> observation) that lets a few imagined
    trajectories dominate an update (MS-Human-700: pre-clip policy gradient norms of 1e2-1e5, ~100 % from 1 % of the
    trajectories). 0: off."""

    # optimization
    lr: float = 2e-4
    lr_decay_iters: int = 250_000
    """Cosine decay to 0.03 x lr over this many iterations."""
    grad_clip: float = 1.0
    policy_grad_clip_value: float = 0.1
    amp: Literal["none", "bf16"] = "bf16"
    """Autocast dtype for MLP matmuls (state integration, FK and rewards stay fp32)."""
    compile: bool = True
    """torch.compile the world-model rollout, imagination step and heads (CUDA only)."""

    # ---- BlinkMimic / InstantTrack options (github.com/fuyichen0725/blinkmimic). The defaults reproduce the runner
    # above; fast_runner_cfg() sets the recipe ----
    pi_ema: float = 0.0
    """Polyak average of the policy weights and normalizers (decay per policy update; InstantTrack: 0.995). The
    average is the deployed policy: checkpoints store it as "pi" (what evaluation and play read) and the live policy as
    "pi_live"; collection keeps using the live policy. 0: off."""
    pi_opt: Literal["adam", "muon"] = "adam"
    """Policy optimizer: Adam, or Muon (Jordan et al. 2024) on the hidden weight matrices of the encoder and decoder at
    Adam's update RMS (same lr; Liu et al. 2025) with Adam on the first / last layers and vectors (optim.py)."""
    wm_opt: Literal["adam", "muon"] = "adam"
    """World-model optimizer: Adam, or the same Muon / Adam split on the world-model MLP."""
    muon_momentum: float = 0.95
    """Momentum of Muon's (Nesterov) update on the hidden matrices, policy and world model (torch.optim.Muon's default)."""
    pi_beta1: float = 0.9
    """Adam beta1 of the policy (SHAC uses 0.7)."""
    fused_adam: bool = False
    """Fused Adam kernels (with no_sync also the device-side skip of a non-finite policy update)."""
    no_sync: bool = False
    """Remove the host-device synchronizations of the update loop: the normalizers' freeze test, the sampling of
    imagination starts (a stable sort instead of nonzero) and the finite-gradient test before the policy step (the
    gradient is zeroed and the step gated on the device: needs fused_adam). Same updates up to floating-point rounding."""
    pi_lr_mult: float = 1.0
    """Policy learning rate = lr x this (world model and critic keep lr)."""
    lr_decay_ret: float = 0.0
    """Progress-triggered decay of pi_lr_mult (0: off): once the logged training return (ep_ret_norm, every
    log_interval iterations) reaches this, the multiplier ramps linearly to 1 over lr_decay_len iterations."""
    lr_decay_len: int = 300
    lr_decay_by: int = 0
    """With lr_decay_ret: start the decay at this iteration at the latest, also if the training return has not reached
    lr_decay_ret (0: no deadline)."""
    empty_cache: bool = True
    """torch.cuda.empty_cache() after every policy update: hands the update's cached blocks to MuJoCo-Warp, which
    allocates outside torch's pool, at the price of a device synchronization and the allocator refilling its cache."""
    wm_structured: bool = False
    """Semi-implicit world model (BlinkMimic): the network predicts the velocity increments and position corrections
    on top of the drift the old and new velocities imply over the step's Euler substeps, instead of the position
    increments themselves (networks.WorldModel)."""
    wm_mirror: float = 0.0
    """Share of each world-model batch replaced by its left-right mirror image (BlinkMimic: 0.5), the physical symmetry
    of the body as data augmentation: states, activations, excitations and foot touch are mirrored with the maps of
    tasks/tracking/mdp/mirror.py, validated on the model at start-up. 0: off."""
    cuda_graph: bool = False
    """Systems (BlinkMimic): capture one world-model update and one policy + critic update (sampling, forward,
    backward, optimizer, EMA) as CUDA graphs after three warm-up calls and replay them from then on. The same updates
    up to floating-point rounding, on another random stream. Needs no_sync, fused_adam, empty_cache off and a CUDA
    device; the learning rate is then a device tensor filled every iteration, and Adam keeps its step on the GPU."""
    collect_graph: bool = False
    """Systems (BlinkMimic's --collect_graph --reset_masked): one collection step (observation, policy and action
    noise, the simulator's substeps including MuJoCo-Warp's solver loop, reward, termination, buffer write, normalizer
    updates, bin statistics and the resets of ended episodes as fixed-size masked updates) runs through a thin
    simulator wrapper (collector.py) in place of mjlab's manager env and is replayed as one CUDA graph after three
    eager warm-up steps: no host synchronization in a step. The same transitions up to floating-point rounding, on
    another random stream. Needs no_sync and the motion command's bin_fix and no observation history;
    the replay buffer's write head lives on the device. On a CPU device the wrapper runs eagerly."""
    overlap_collect: bool = False
    """Systems: each iteration's collection runs on a second CUDA stream, concurrently with that iteration's world-model
    and policy updates, once the collection and update graphs are captured (iterations before that run serially).
    The collector acts with its own copy of the policy, refreshed from the live policy at the start of every iteration
    (the parameters it acts with are the same as without overlap), and owns the observation and world-model
    normalizer statistics, copied to the trainer's at the start of every iteration; it draws from its own random
    state. The updates sample the buffer as it stood at the start of the iteration, without the rows the collection
    is overwriting: the newest steps reach them one iteration later. Needs collect_graph and cuda_graph on a GPU."""
    overlap_wm: bool = False
    """Systems (InstantTrack v1.3's --overlap_wm): once both update graphs are captured, the world-model updates run on
    their own stream concurrently with the policy update; imagination steps a copy of the world model (parameters and
    normalizers) taken at the start of each iteration's updates, one update round older. The world-model graph is
    captured on its own stream with its own random state. Needs cuda_graph."""
    max_hours: float = 0.0
    """Stop after this much training wall-clock (the runner's clock: compilation, graph capture and checkpoints
    included, process start-up and data loading excluded), saving a final checkpoint. 0: no limit."""
    finetune: bool = False
    """Fine-tuning from a baseline checkpoint (--agent.resume or finetune.py): load its networks, normalizers and
    optimizer states, but restart the iteration count, the learning-rate schedule and the training clock, and skip
    the world-model-only warm-up (the world model is trained)."""
    accept_contract_mismatch: bool = False
    """Load a checkpoint whose task contract (observation layout, actuators, model and version, timing) differs from
    this env's. Off by default: equal widths do not make two observation layouts compatible. Set it only for an
    intentional transfer, e.g. fine-tuning onto a new model version."""

    # ---- XMimic systems options (github.com/Sentient-X/xmimic): the same updates up to floating-point rounding ----
    kin_unbind: bool = True
    """Forward kinematics read the joint coordinates through one unbind instead of a select per joint, so the backward
    pass stacks the joint gradients once instead of scattering each into a zero-filled tensor."""
    hoist_goal: bool = False
    """Imagination: the goals of all imagined steps and their encoder latents in one batched pass before the loop
    (they depend on the reference frame only), instead of once per step."""
    muon_batched: bool = False
    """Muon's Newton-Schulz iteration on the same-shape hidden matrices as batched GEMMs (cuda_graph only)."""
    gc_freeze: bool = True
    """gc.freeze() after setup: the garbage collector skips the objects of setup and compilation."""


def fast_runner_cfg(**overrides: object) -> SuperMimicRunnerCfg:
    """muscle-supermimic's fast recipe (``scripts/train_blink.sh`` at 8a6244b, 50 Hz control).

    BlinkMimic / InstantTrack's method changes as they transfer to muscles: the policy EMA (0.995), Muon on the hidden
    matrices of policy and world model, no host synchronization, fused Adam, learning rate 4e-4, a 3 x 1024 world model
    with 5 batches per iteration, an imagination batch of 2048, the semi-implicit world model and left-right mirror
    augmentation of half of each world-model batch. Changed for muscles: bootstrapped values clamped to the attainable
    maximum, a 200-iteration world-model warm-up and the policy learning rate at 1x. muscle-supermimic's own: Beta
    exploration noise, the updates and the collection as CUDA graphs with the collection overlapped, no
    rematerialization of the imagined steps, and 50 Hz with every step-counted setting rescaled (horizon 16 = 0.32 s,
    gamma 0.98, lambda 0.90, a 500-step buffer, 4-step world-model windows, 1.25 simulator steps per iteration, noise
    0.1 / sqrt(2)). The env side (50 Hz, goal stride 10, 1024 envs, bin_fix, fail_reset 0.3) is the registered task's.
    On a CPU device the runner turns the CUDA-graph options off.

    Args:
        **overrides: Field values replacing the recipe's.

    Returns:
        The runner config.
    """
    recipe: dict[str, object] = dict(
        experiment_name="supermimic_fullbody", wandb_project="myosuite_supermimic", pi_ema=0.995, no_sync=True, fused_adam=True, lr=4e-4, wm_batches=5,
        wm_hidden=(1024, 1024, 1024), pi_opt="muon", wm_opt="muon", wm_only_iters=200, value_clip=True, pi_batch=2048,
        pi_lr_mult=1.0, empty_cache=False, wm_structured=True, wm_mirror=0.5, cuda_graph=True, collect_graph=True,
        overlap_collect=True, remat_imagination=False, action_noise_dist="beta", save_interval=250,
        max_iterations=20000, horizon=16, gamma=0.98, lam=0.9, buffer_steps=500, wm_window=4, utd=0.8,
        action_noise=0.0707, hoist_goal=True, muon_batched=True,
    )  # fmt: skip
    recipe.update(overrides)
    return SuperMimicRunnerCfg(**recipe)  # type: ignore[arg-type]
