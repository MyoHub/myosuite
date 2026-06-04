# SAR Backend Benchmark

Compares the **Synergy-based Action Representation (SAR)** pipeline across
physics backends and training algorithms.  A deterministic rollout (Phase A)
checks numerical parity; a training run (Phase B) checks convergence.

---

## Backends at a glance

| `--backends` flag | Physics engine | Train algorithm | GPU required? |
|---|---|---|---|
| `cpu` | MuJoCo C++ float64 | SB3 SAC (CPU) | No |
| `mjx_xla` | JAX/XLA float32 | SB3 SAC (CPU-side policy) | No (GPU speeds up physics) |
| `mjx_warp` | MuJoCo Warp float32 | SB3 SAC (CPU-side policy) | **Yes** |
| `mjlab` | MuJoCo Warp float32 (Isaac Lab) | RSL-RL PPO (fully GPU) | No (GPU strongly recommended) |

> **Algorithm note**: `cpu`, `mjx_xla`, `mjx_warp` all use SB3 SAC (off-policy,
> single environment).  `mjlab` uses RSL-RL PPO (on-policy, vectorised,
> all tensors on GPU — no CPU round-trips in the training hot path).  Reward
> curves are not directly comparable step-for-step, but both families should
> converge if SAR is working.

---

## Prerequisites

### Always needed
```bash
pip install stable-baselines3 scikit-learn
```

### MJX XLA (CPU or GPU)
```bash
pip install jax jaxlib     # CPU-only JAX
# or for GPU:
pip install jax[cuda12]    # JAX with CUDA 12 support
```

### MJX Warp (GPU only)
```bash
pip install warp-lang      # NVIDIA Warp — requires a CUDA GPU
```

### mjlab (GPU strongly recommended)
```bash
pip install rsl-rl         # RSL-RL PPO runner
# Isaac Lab / mjlab must be installed separately — see docs/mjlab_setup.md
```

---

## Running the benchmark

All examples are run from the **repo root**.

### 1. CPU only

Pure MuJoCo C++, no GPU needed.

```bash
python benchmarks/sar_backends/run_benchmark.py \
    --backends cpu \
    --steps 50000
```

---

### 2. MJX XLA on CPU

Forces JAX to use the CPU XLA backend (slower, but useful on machines without
a GPU or to test numerical parity with the C++ backend).

```bash
JAX_PLATFORM_NAME=cpu python benchmarks/sar_backends/run_benchmark.py \
    --backends mjx_xla \
    --steps 50000
```

---

### 3. MJX XLA on GPU

JAX auto-selects CUDA if available.  Nothing extra needed beyond having
`jax[cuda12]` installed.

```bash
python benchmarks/sar_backends/run_benchmark.py \
    --backends mjx_xla \
    --steps 50000
```

To pin to a specific GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/sar_backends/run_benchmark.py \
    --backends mjx_xla \
    --steps 50000
```

---

### 4. MJX Warp on GPU

Warp always runs on GPU.  Enable with `MYOSUITE_GPU_TESTS=1` to signal that a
GPU is present (the script skips Warp silently otherwise).

```bash
MYOSUITE_GPU_TESTS=1 python benchmarks/sar_backends/run_benchmark.py \
    --backends mjx_warp \
    --steps 50000
```

---

### 5. mjlab on GPU (recommended)

Physics (MuJoCo Warp) and the full RSL-RL PPO training loop (policy, SAR
inverse transform, rollout buffer, update) all stay on GPU.

```bash
MYOSUITE_GPU_TESTS=1 python benchmarks/sar_backends/run_benchmark.py \
    --backends mjlab \
    --steps 50000
```

To pin the GPU:
```bash
CUDA_VISIBLE_DEVICES=0 MYOSUITE_GPU_TESTS=1 \
    python benchmarks/sar_backends/run_benchmark.py \
    --backends mjlab \
    --steps 50000
```

---

### 6. mjlab on CPU (fallback only)

The mjlab runner falls back to CPU automatically when no CUDA device is found.
This is **not recommended** for real training — RSL-RL PPO on CPU with the
full Isaac Lab env stack is very slow.

```bash
CUDA_VISIBLE_DEVICES="" python benchmarks/sar_backends/run_benchmark.py \
    --backends mjlab \
    --steps 5000
```

---

### 7. All backends at once (GPU machine)

```bash
MYOSUITE_GPU_TESTS=1 python benchmarks/sar_backends/run_benchmark.py \
    --backends cpu mjx_xla mjx_warp mjlab \
    --steps 100000 \
    --output-dir ./benchmark_results \
    --seed 42
```

---

### 8. CPU vs MJX-XLA parity check only (no training)

Phase A runs a 200-step deterministic rollout and compares qpos / reward
across backends.  Useful for catching numerical divergence without the cost of
a training run.

```bash
python benchmarks/sar_backends/run_benchmark.py \
    --backends cpu mjx_xla \
    --skip-phase-b \
    --rollout-steps 500
```

---

## All CLI flags

```
usage: run_benchmark.py [-h]
                        [--steps STEPS]
                        [--output-dir PATH]
                        [--sar-dir PATH]
                        [--backends {cpu,mjx_xla,mjx_warp,mjlab} ...]
                        [--seed SEED]
                        [--skip-phase-a]
                        [--skip-phase-b]
                        [--rollout-steps ROLLOUT_STEPS]

--steps N           Total training timesteps per backend (Phase B). Default: 10000.
--output-dir PATH   Root output directory. Default: ./sar_benchmark_results
--sar-dir PATH      Path to pretrained SAR pkl files (ica.pkl, pca.pkl,
                    normalizer.pkl). Default: uses bundled files.
--backends LIST     Space-separated subset of: cpu mjx_xla mjx_warp mjlab.
                    Default: all four.
--seed N            Random seed. Default: 0.
--skip-phase-a      Skip the deterministic rollout comparison.
--skip-phase-b      Skip the training comparison.
--rollout-steps N   Steps for Phase A rollout. Default: 200.
```

---

## Output structure

```
<output-dir>/
├── benchmark.log
├── phase_a_rollout/           # Phase A — deterministic parity
│   ├── cpu_rollout.csv
│   ├── mjx_xla_rollout.csv
│   ├── mjx_warp_rollout.csv
│   ├── mjlab_rollout.csv
│   └── divergence_report.txt
└── phase_b_training/          # Phase B — training
    ├── cpu/
    │   ├── training_log.csv   # per-episode: timestep, ep_reward, vel_reward, …
    │   └── steps_log.csv      # per-step diagnostics (sampled)
    ├── mjx_xla/
    │   ├── training_log.csv
    │   └── steps_log.csv
    ├── mjx_warp/
    │   └── training_log.csv
    ├── mjlab/
    │   ├── training_log.csv   # per-PPO-iteration metrics
    │   └── rslrl_logs/        # RSL-RL tensorboard logs
    ├── comparison_reward.png
    ├── comparison_components.png
    └── comparison_physics.png
```

After running the benchmark, you can render **only the solutions** (trained
policies) as videos—one per backend that has a checkpoint—with:

```bash
python benchmarks/sar_backends/render_benchmark_results.py
```

By default this uses ``sar_benchmark_results/phase_b_training`` and writes
``render_rollout.mp4`` into each backend's run directory. Options:

- ``--results-dir PATH`` — root benchmark output (default: ./sar_benchmark_results)
- ``--sar-dir PATH`` — SAR pkl files (default: myosuite/agents/SAR_pretrained/locomotion)
- ``--steps N`` — frames to render (default: 500)
- ``--seed N`` — rollout seed (default: 42)
- ``--backends cpu mjx_xla mjx_warp mjlab`` — subset to render (default: all with a saved solution)

---

## Quick decision guide

| Goal | Command |
|---|---|
| Sanity check on CPU-only machine | `--backends cpu mjx_xla` |
| Check Warp physics parity (GPU) | `MYOSUITE_GPU_TESTS=1 --backends cpu mjx_warp --skip-phase-b` |
| Full GPU throughput comparison | `MYOSUITE_GPU_TESTS=1 --backends mjx_xla mjx_warp mjlab` |
| Reproduce a CI failure | `--backends cpu mjx_xla --seed 0 --steps 10000` |
| mjlab GPU training only | `MYOSUITE_GPU_TESTS=1 --backends mjlab --skip-phase-a` |
| Render solutions (trained policies) to video | `python benchmarks/sar_backends/render_benchmark_results.py` |
