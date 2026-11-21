# DreamerV3 world-model training for MyoSuite

Model-based reinforcement learning with DreamerV3 on registered MyoSuite tasks across all three
simulation paths, plus standalone PyTorch / TorchScript / ONNX deployment (with MyoSuite physical
deployment manifests) and pinned Hugging Face checkpoint resolution:

1. **CPU Gymnasium (`DreamerEnv`, `dreamerv3-train`)** — consumes `myosuite.make_env` in
   proprioceptive or offscreen `64x64` RGB visual mode, with `--feature` muscle-condition wrappers.
2. **GPU `mjlab` (`MjlabDriver`, `dreamerv3-train-mjlab`)** — consumes MyoSuite's registered GPU
   MuJoCo Warp twins (`ManagerBasedRlEnv`), tracking peak deterministic checkpoints (`best_ckpt`)
   that evaluate zero-shot on CPU `make_env`.
3. **GPU MJX (`MJXDriver`, `dreamerv3-train-mjx`)** — steps experimental JAX `MyoMjxEnvBase` tasks
   inside `jax.jit(jax.vmap(...))` while preserving terminal observations prior to reset.
4. **Standalone TorchScript & ONNX Export (`DreamerV3TorchPolicy`, `dreamerv3-export`)** —
   reconstructs the encoder, block-GRU RSSM posterior, and bounded-normal actor directly from saved
   checkpoint arrays (`atol < 1e-5` vs. JAX) and emits `<policy>.onnx.manifest.json` sidecars for
   strict `myosuite.utils.export_onnx.verify_onnx_on_cpu` validation (`allow_provenance_mismatch=False`).

## Attribution

DreamerV3 is the method of

> Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap. **Mastering Diverse Domains
> through World Models.** *Nature* 640, 647–653 (2025).
> [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) ·
> reference implementation: [github.com/danijar/dreamerv3](https://github.com/danijar/dreamerv3) (MIT License)

This is not an original MyoSuite learning algorithm. The RSSM world model, actor-critic imagination
objectives, replay buffer, and size presets (`size1m`–`size50m`) come from `danijar/dreamerv3` (whose
MIT license notice is preserved in `LICENSE-dreamerv3.txt`).

| Component | Source |
|---|---|
| RSSM world model, symlog/twohot heads, actor-critic imagination, `configs.yaml` presets | Upstream `danijar/dreamerv3` (MIT License, imported externally via `dreamerv3` / `embodied`). |
| `myosuite_dreamerv3/env.py` (`DreamerEnv`, `wrap_dreamer_env`) | New MyoSuite adapter wrapping `myosuite.make_env` (5-tuple Gymnasium step, `EnvConfig.features` muscle wrappers, offscreen `mujoco.Renderer` RGB frames, `solved` / `rwd_dense` / `rwd_sparse` logging). |
| `myosuite_dreamerv3/mjlab_driver.py` (`MjlabDriver`, `train_mjlab`) | New MyoSuite batched GPU `mjlab` (MuJoCo Warp + PyTorch) driver handling same-step auto-reset, `jax.prealloc=False` GPU memory sharing, and in-place `bfloat16` JAX recurrent carry selection. |
| `myosuite_dreamerv3/mjx_driver.py` (`MJXDriver`) | New MyoSuite batched GPU JAX driver (`jax.jit(jax.vmap(...))`) with `jax.lax.cond` step-vs-reset staging so terminal observations (`is_last=True`, `is_terminal`) are preserved before episode reset. |
| `myosuite_dreamerv3/export.py` (`DreamerV3TorchPolicy`, `export_dreamerv3_onnx`, `write_onnx_deployment_manifest`) | New pure-PyTorch mirror, single-file `.onnx` + `.onnx.manifest.json` provenance sidecar exporter, and TorchScript `.pt` exporter. |
| `myosuite_dreamerv3/hf.py` (`resolve_dreamerv3_checkpoint`, `stage_hf_release_bundle`) | New Hugging Face Hub helpers enforcing pinned `@<commit_or_tag>` revisions (rejecting moving branches such as `main`). |
| `myosuite_dreamerv3/compat.py`, `train.py`, `train_mjx.py`, `benchmark.py` | New runtime compatibility shims (JAX >= 0.8 / `0.12.x`, Optax, CUDA XLA, NumPy `2.5.5`, Windows POSIX checkpoint paths), `best_ckpt` tracking, CLI entrypoints, and multi-seed PPO benchmark harness. |

## Layout

- `myosuite_dreamerv3/env.py` — CPU `DreamerEnv` (`embodied.Env`) consuming `myosuite.make_env`.
- `myosuite_dreamerv3/mjlab_driver.py` — `dreamerv3-train-mjlab` CLI and batched GPU `MjlabDriver` for registered `mjlab` twins.
- `myosuite_dreamerv3/mjx_driver.py` — Batched GPU `MJXDriver` for experimental MyoSuite MJX tasks.
- `myosuite_dreamerv3/export.py` — `dreamerv3-export` CLI and JAX-free PyTorch / TorchScript / ONNX + deployment-manifest export.
- `myosuite_dreamerv3/hf.py` — Pinned `hf://owner/repo@<rev>/...` resolution and baseline bundle staging.
- `myosuite_dreamerv3/compat.py` — Namespace-shadowing guard and JAX / Optax / CUDA XLA / Windows shims.
- `myosuite_dreamerv3/train.py` — `dreamerv3-train` CLI for CPU proprioceptive/visual training and deterministic evaluation.
- `myosuite_dreamerv3/train_mjx.py` — `dreamerv3-train-mjx` CLI and shared batched-driver training loop with `best_ckpt` tracking.
- `myosuite_dreamerv3/benchmark.py` — `dreamerv3-bench` CLI comparing `PPO` (`mjlab` RSL-RL or SB3) and `DreamerV3` with milestone learning curves.

## Install

Use a dedicated virtual environment so DreamerV3's JAX / `elements` / `ninjax` stack stays isolated
from other trainers:

```bash
pip install -e ".[mjlab,rl]"
git clone https://github.com/danijar/dreamerv3.git ../dreamerv3
export PYTHONPATH="$(cd ../dreamerv3 && pwd):${PYTHONPATH:-}"
pip install -e "training/dreamerv3[mjlab,rl,onnx,hf,test]"
```

For CUDA world-model training on Linux, install `jax[cuda12]`.

## Use

```bash
# 1. GPU mjlab twin + CUDA DreamerV3 size12m with ONNX + deployment manifest export
dreamerv3-train-mjlab --env-id myoElbowPose1D6MRandom-v0 --num-envs 64 --size size12m \
    --steps 200000 --export-onnx logs/elbow_dreamerv3.onnx

# 2. Export a local or pinned Hugging Face checkpoint to ONNX + TorchScript and verify on CPU
dreamerv3-export --checkpoint logs/dreamerv3_run --onnx policy.onnx --torchscript policy.pt \
    --verify-env myoElbowPose1D6MRandom-v0
python -m myosuite.utils.export_onnx verify --onnx policy.onnx --env myoElbowPose1D6MRandom-v0

# 3. CPU Gymnasium task with offscreen 64x64 RGB observations and muscle fatigue
dreamerv3-train --env-id myoElbowPose1D6MRandom-v0 --size size25m --vision --feature fatigue

# 4. Reproducible 3-seed GPU benchmark with step & wall-time learning curves
dreamerv3-bench --envs-list myoElbowPose1D6MRandom-v0 myoHandPoseRandom-v0 myoLegWalk-v0 \
    --seeds 0 1 2 --steps 36864 --eval-episodes 20 --size size12m --platform cuda --curve-points 4 \
    --ppo-backend mjlab --ppo-num-envs 256 --dreamer-backend mjlab --num-envs 64 --train-ratio 48
```

## Benchmark against PPO

All trained `size12m` checkpoints (`agent.pkl`, TorchScript `.pt`, and stateful `.onnx` +
`.onnx.manifest.json` across `seeds = 0, 1, 2`) and raw evaluation / learning-curve JSON logs are
published on Hugging Face at [`maco018/myosuite-dreamerv3`](https://huggingface.co/maco018/myosuite-dreamerv3/tree/f1d6f8634f2e5fc1208cad9eafec39f67b0aa843),
pinned at tag `v0.1.0` / commit `f1d6f8634f2e5fc1208cad9eafec39f67b0aa843`:

```python
from myosuite_dreamerv3 import evaluate_exported_onnx_stateful, resolve_dreamerv3_checkpoint

onnx_path = resolve_dreamerv3_checkpoint(
    "hf://maco018/myosuite-dreamerv3@f1d6f8634f2e5fc1208cad9eafec39f67b0aa843/"
    "checkpoints/myoElbowPose1D6MRandom-v0/seed0.onnx"
)
print(evaluate_exported_onnx_stateful(onnx_path, "myoElbowPose1D6MRandom-v0", episodes=20))
```

**Difference summary:** Across all four evaluated tasks (`seeds = 0, 1, 2` on an NVIDIA GeForce RTX
5090, evaluated deterministically zero-shot on CPU `myosuite.make_env`), `DreamerV3` (`size12m`,
`mjlab`) achieves higher sample efficiency and peak task return than registered `mjlab` RSL-RL `PPO`
on 6-muscle `myoElbowPose1D6MRandom-v0` (`100.0 ± 0.0%` vs. `75.0 ± 8.2%` solved; `735.71 ± 24.86`
vs. `281.25 ± 121.26` return at `36,864` steps), under `--feature fatigue` (`100.0 ± 0.0%` vs.
`66.7 ± 2.4%` solved; `732.57 ± 18.91` vs. `226.56 ± 11.51` return), on 39-muscle
`myoHandPoseRandom-v0` (`-276.76 ± 15.28` vs. `-300.81 ± 2.56` return at `294,912` steps), and on
80-muscle bipedal `myoLegWalk-v0` (`231.43 ± 41.67` vs. `211.49 ± 15.11` return at `196,608` steps),
whereas massively parallel on-policy `mjlab` `PPO` trains `7.8×`–`15.8×` faster in wall-clock time
(`3.7–22.2 s` vs. `41.3–351.6 s`).

### 1. Flagship `size12m` GPU Summary (`mjlab` MuJoCo Warp + RTX 5090, Seeds `0, 1, 2`)

Hardware: WSL2 Linux, NVIDIA GeForce RTX 5090 (32 GiB), 24 CPU threads, Python `3.13.18`,
`numpy 2.5.5`, `gymnasium 1.4.0`, `jax 0.12.1`, `torch 2.15.0+cu132`.

| Task (`env_id`) | Method | Env Steps | CPU `make_env` Return | CPU Solved Rate | Wall Time (s) |
|---|---|---:|---:|---:|---:|
| `myoElbowPose1D6MRandom-v0` | **`DreamerV3` (`size12m`, `mjlab`)** | 36,864 | **`735.71 ± 24.86`** | **`100.0 ± 0.0%`** | `41.3 ± 1.3` |
| `myoElbowPose1D6MRandom-v0` | `PPO` (`mjlab` / RSL-RL) | 36,864 | `281.25 ± 121.26` | `75.0 ± 8.2%` | **`3.7 ± 0.3`** |
| `myoHandPoseRandom-v0` | **`DreamerV3` (`size12m`, `mjlab`, `best_ckpt`)** | 294,912 | **`-276.76 ± 15.28`** (`-280.93 ± 14.75` final) | `0.0 ± 0.0%` | `351.6 ± 33.9` |
| `myoHandPoseRandom-v0` | `PPO` (`mjlab` / RSL-RL) | 294,912 | `-300.81 ± 2.56` | `0.0 ± 0.0%` | **`22.2 ± 6.4`** |
| `myoLegWalk-v0` | **`DreamerV3` (`size12m`, `mjlab`, `best_ckpt`)** | 196,608 | **`231.43 ± 41.67`** (`165.17 ± 10.10` final) | `0.0 ± 0.0%` | `266.3 ± 7.5` |
| `myoLegWalk-v0` | `PPO` (`mjlab` / RSL-RL) | 196,608 | `211.49 ± 15.11` | `0.0 ± 0.0%` | **`20.0 ± 1.2`** |

### 2. Learning Curves (Environment Steps & Wall-Clock Time, Mean ± Std Over Seeds `0, 1, 2`)

#### `myoElbowPose1D6MRandom-v0` (`0` → `36,864` Steps, `size12m` `train_ratio=48` vs. `mjlab` PPO)

| Env Steps | `DreamerV3 (size12m)` Wall Time (s) | `DreamerV3` Return (`Solved`) | `mjlab` PPO Wall Time (s) | `mjlab` PPO Return (`Solved`) |
|---:|---:|---:|---:|---:|
| `0` | `0.0` | `14.93 ± 10.49` (`43.3%`) | `0.0` | `5.24 ± 4.62` (`33.3%`) |
| `12,288` | `14.9` | `40.51 ± 44.09` (`40.0%`) | `1.5` | `66.69 ± 31.33` (`45.0%`) |
| `24,576` | `28.1` | `112.51 ± 21.21` (`50.0%`) | `2.6` | `229.74 ± 28.13` (`56.7%`) |
| `36,864` | `41.3` | **`735.71 ± 24.86` (`100.0%`)** | `3.7` | `281.25 ± 121.26` (`75.0%`) |

#### `myoHandPoseRandom-v0` (`0` → `294,912` Steps, `size12m` `train_ratio=48` vs. `mjlab` PPO)

| Env Steps | `DreamerV3 (size12m)` Wall Time (s) | `DreamerV3` Return | `mjlab` PPO Wall Time (s) | `mjlab` PPO Return |
|---:|---:|---:|---:|---:|
| `0` | `0.0` | `-316.51 ± 0.85` | `0.0` | `-320.41 ± 2.37` |
| `73,728` | `85.0` | `-366.68 ± 11.86` | `6.1` | `-316.73 ± 2.67` |
| `147,456` | `174.4` | `-316.74 ± 24.88` | `11.4` | `-308.40 ± 5.80` |
| `221,184` | `263.0` | `-290.95 ± 15.55` | `16.8` | `-303.23 ± 4.77` |
| `294,912` | `351.6` | **`-280.93 ± 14.75`** (**`-276.76 ± 15.28`** `best_ckpt`) | `22.2` | `-300.81 ± 2.56` |

#### `myoLegWalk-v0` (`0` → `196,608` Steps, `size12m` `train_ratio=32` vs. `mjlab` PPO)

| Env Steps | `DreamerV3 (size12m)` Wall Time (s) | `DreamerV3` Return | `mjlab` PPO Wall Time (s) | `mjlab` PPO Return |
|---:|---:|---:|---:|---:|
| `0` | `0.0` | `214.59 ± 3.87` | `0.0` | `197.59 ± 15.42` |
| `49,152` | `69.3` | `135.96 ± 11.45` | `6.1` | `205.88 ± 22.14` |
| `98,304` | `136.3` | **`244.68 ± 44.00`** | `10.8` | `205.41 ± 14.94` |
| `147,456` | `201.7` | `148.16 ± 89.52` | `15.4` | `212.46 ± 11.67` |
| `196,608` | `266.3` | `165.17 ± 10.10` (**`231.43 ± 41.67`** `best_ckpt`) | `20.0` | `211.49 ± 15.11` |

### 3. Modified Task Condition (`--feature fatigue`, Reported Separately) & CPU Smoke Reference

#### `myoElbowPose1D6MRandom-v0` with `FatigueWrapper` (`--feature fatigue` on GPU `mjlab`, `36,864` Steps)

| Env Steps | `DreamerV3 (size12m)` Wall Time (s) | `DreamerV3` Return (`Solved`) | `mjlab` PPO Wall Time (s) | `mjlab` PPO Return (`Solved`) |
|---:|---:|---:|---:|---:|
| `0` | `0.0` | `14.63 ± 12.91` (`43.3%`) | `0.0` | `5.30 ± 4.55` (`33.3%`) |
| `12,288` | `15.3` | `40.68 ± 44.14` (`40.0%`) | `3.2` | `39.48 ± 20.47` (`43.3%`) |
| `24,576` | `29.4` | `182.44 ± 29.73` (`55.0%`) | `4.4` | `192.12 ± 45.69` (`55.0%`) |
| `36,864` | `43.5` | **`732.57 ± 18.91` (`100.0 ± 0.0%`)** | `5.6` | `226.56 ± 11.51` (`66.7 ± 2.4%`) |

#### CPU Smoke Reference (`stable-baselines3` PPO vs. `debug` DreamerV3, Unmodified Tasks, `2,048` Steps)

| Task (`env_id`) | Method | Env Steps | Return (mean ± std) | Solved Rate (mean ± std) | Wall Time (s) | Peak RSS (MB) |
|---|---|---:|---:|---:|---:|---:|
| `myoElbowPose1D6MRandom-v0` | `PPO` (`stable-baselines3`) | 2,048 | `84.22 ± 22.81` | `50.0 ± 8.2%` | `1.7 ± 0.5` | `917.5` |
| `myoElbowPose1D6MRandom-v0` | `DreamerV3` (`debug` world model) | 2,048 | `155.09 ± 15.59` | `80.0 ± 8.2%` | `11.3 ± 0.1` | `1,070.8` |
| `myoHandPoseRandom-v0` | `PPO` (`stable-baselines3`) | 2,048 | `-312.15 ± 2.27` | `0.0 ± 0.0%` | `1.9 ± 0.0` | `1,492.0` |
| `myoHandPoseRandom-v0` | `DreamerV3` (`debug` world model) | 2,048 | `-349.00 ± 4.28` | `0.0 ± 0.0%` | `19.7 ± 0.1` | `1,635.9` |

## Tests

```bash
PYTHONPATH=../dreamerv3 pytest training/dreamerv3/tests -v
```
