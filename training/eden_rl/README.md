# `myosuite-eden-rl` (`training/eden_rl`)

Self-contained reinforcement learning extensions adapted from [Eden](https://github.com/embodied-ai-nexus/Eden) (Apache-2.0), [SimBaV2](https://github.com/DAVIAN-Robotics/SimbaV2) (Apache-2.0), [HIMLoco](https://github.com/InternRobotics/HIMLoco) / [OpenHomie](https://github.com/OpenRobotLab/HomieRL) (MIT), [Muon](https://github.com/KellerJordan/Muon) (MIT), and [AMP](https://arxiv.org/abs/2104.02180) for high-dimensional musculoskeletal control in MyoSuite (`myoHand`, `myoArm`, `myoLeg`, and `myoFullBody` — up to 290–700 Hill-type muscles).

Per [`docs/wiki/training-guidelines.md`](../../docs/wiki/training-guidelines.md), method-specific architectures, optimizers, self-supervised estimators, and adversarial replay buffers live here in `training/eden_rl/` so core `myosuite` stays lightweight and free of method-specific learning code. Importing `eden_rl` registers `rebuild_simba_trunk` with `myosuite.utils.rslrl_policy.register_actor_trunk_builder`, and `eden_rl.export_simba_to_onnx` exports trained checkpoints to self-contained `.onnx` graphs that play back in core MyoSuite without `eden_rl` installed.

## Components

| Module | Adapted Upstream Method | MyoSuite-Specific Integration |
| :--- | :--- | :--- |
| `eden_rl.simba` | `SimBa`, `GroupEncoder`, `SimBaModel` — SimBaV2 hyperspherical normalization (`HyperEmbedder` + `HyperLERPBlock`) from Lee et al. (2025) & Eden | `rebuild_simba_trunk`, `register_simba_rslrl_loader` hook for `myosuite.utils.rslrl_policy`, `load_simba_policy`, and `export_simba_to_onnx` |
| `eden_rl.muon_ppo` | `MuonAdamW`, `MuonPPO` — 2-D `torch.optim.Muon(..., adjust_lr_fn="match_rms_adamw")` + 1-D `AdamW` pairing from Jordan et al. (2024), Weave & Eden | Single-schedule LR synchronization across high-dimensional muscle policy heads and RSL-RL >= 5 runner config wiring |
| `eden_rl.him` | `sinkhorn`, `HIMEstimator`, `him_symmetry_losses` — SwAV Sinkhorn-Knopp history encoder from Long et al. (ICLR 2024) & Eden | Generalized `target_dim` to regress privileged physiological vectors (3CC-r muscle fatigue, sarcopenia peak-force factors, external loads) + single-input ONNX-exportable `HIMActor` |
| `eden_rl.amp` | `AmpMotionSampler`, `AmpDiscriminator`, `AmpReplayBuffer`, `FrameNormalizer`, `AmpPPO` — spectral-norm LSGAN + WGAN-GP motion prior from Peng et al. (SIGGRAPH 2021) & Eden | Direct loading of `myoMimic-v0` refitted `.npz` trajectory clips (`qpos` slices + `fps`/`dt`) and in-memory tensor clips without external asset managers |
| `eden_rl.genesis_bridge` | *(New MyoSuite glue)* | `GenesisMuscleBridge` driving stripped joint-torque `genesis-world` / `Eden` `RigidEntity` skeletons with exact `myo_sim` / `myosuite.physics.hill_muscle` curves via `entity.control_dofs_force(...)` |
| `eden_rl.train` | *(New MyoSuite glue)* | CLI (`myosuite-eden-train`) and runner config builder wrapping `myosuite.make_env(..., backend="mjlab")` |

## Installation

```bash
# Standalone PyTorch modules + Genesis muscle bridge (requires torch>=2.10):
pip install -e training/eden_rl

# With MyoSuite mjlab + RSL-RL >= 5 runner integration:
pip install -e ".[mjlab]"
pip install -e "training/eden_rl[rsl]"
```

## Quick Start

```bash
# Train myoLegWalk-v0 on mjlab with SimBaV2 + MuonPPO:
myosuite-eden-train --env myoLegWalk-v0 --backbone simba --algo muon_ppo
```

```python
# Export a trained SimBa checkpoint to a standalone MyoSuite ONNX policy:
from eden_rl import export_simba_to_onnx

export_simba_to_onnx(
    "logs/rsl_rl/myoLegWalk-v0/model_1000.pt",
    "policy.onnx",
    obs_dim=36,
    act_dim=80,
)
```

## Benchmark Against Registered `myo_ppo_runner_cfg` (`mjlab` PPO)

**One-sentence summary:** Compared side by side against the canonical registered `myo_ppo_runner_cfg` (`scripts/train_mjlab.py`) across seeds `0, 1, 2` at an identical `184,320`-step GPU `mjlab` budget (`num_envs = 256`), `SimBaV2 + MuonPPO` (`hidden_dim=256, num_blocks=2, latent_dim=64`) improves final CPU `make_env` return on **80-muscle `myoLegStandRandom-v0`** by **`+135.31`** (`676.82 ± 6.68` vs. `541.51 ± 33.22`, with **`5.0×` lower seed variance**, `91.7% ± 2.4%` solved), improves 3D fingertip reach solved rate on **`myoFingerReachRandom-v0`** (`35.0% ± 4.1%` vs. `33.3% ± 6.2%`), achieves **`+98.8` higher early return** at `61.4k` steps on **`myoFingerPoseRandom-v0`** (`68.4` vs. `-30.4`, `85.0%` final solved), and reaches **`100.0% ± 0.0%`** solved on **216-muscle `myoTorsoPoseFixed-v0`** at `1.8×`–`2.5×` wall-clock cost due to Newton-Schulz orthogonalization in `MuonAdamW`.

- **Hardware:** `NVIDIA GeForce RTX 4070 Ti SUPER` (16 GiB, `torch 2.15.0+cu132`, `mujoco-warp 3.11.0`, `mjlab 1.6.1`, `rsl-rl-lib 5.4.2`).
- **Protocol:** `num_envs = 256`, `30` iterations × `24` steps (`184,320` env steps), seeds `0, 1, 2`, evaluated deterministically over `20` seeded episodes (`seed = 10000..10019`) on unmodified CPU `make_env(env_id)`.

### Final Performance at `184,320` Environment Steps (Mean ± Std over Seeds `0, 1, 2`)

| Task (`env_id`) | Method | CPU `make_env` Return | CPU Solved Rate | Wall-Clock (s) | Peak VRAM (MB) |
|---|---|---:|---:|---:|---:|
| `myoFingerPoseRandom-v0` (5 muscles, 4 DOFs) | `mjlab` PPO (`myo_ppo_runner_cfg`) | **243.54 ± 15.83** | **96.7% ± 2.4%** | **15.34 ± 0.37** | **22.9** |
| `myoFingerPoseRandom-v0` (5 muscles, 4 DOFs) | `SimBaV2 + MuonPPO` | `192.24 ± 140.01` | `85.0% ± 17.8%` | `38.15 ± 0.14` | `92.8` |
| `myoFingerReachRandom-v0` (5 muscles, 3D reach) | `mjlab` PPO (`myo_ppo_runner_cfg`) | **233.97 ± 62.23** | `33.3% ± 6.2%` | **19.05 ± 0.64** | **25.6** |
| `myoFingerReachRandom-v0` (5 muscles, 3D reach) | `SimBaV2 + MuonPPO` | `165.74 ± 30.49` | **35.0% ± 4.1%** | `42.45 ± 0.91` | `93.5` |
| `myoLegStandRandom-v0` (80 muscles, 3D balance) | `mjlab` PPO (`myo_ppo_runner_cfg`) | `541.51 ± 33.22` | **91.7% ± 4.7%** | **28.55 ± 0.51** | **40.2** |
| `myoLegStandRandom-v0` (80 muscles, 3D balance) | `SimBaV2 + MuonPPO` | **676.82 ± 6.68** | **91.7% ± 2.4%** | `50.15 ± 0.05` | `110.1` |
| `myoTorsoPoseFixed-v0` (216 muscles, spine) | `mjlab` PPO (`myo_ppo_runner_cfg`) | **-88.36 ± 13.79** | **100.0% ± 0.0%** | **18.27 ± 0.22** | **62.0** |
| `myoTorsoPoseFixed-v0` (216 muscles, spine) | `SimBaV2 + MuonPPO` | `-111.80 ± 2.97` | **100.0% ± 0.0%** | `40.53 ± 0.24` | `175.5` |

### Learning Curves vs. Environment Steps & Wall-Clock Time (3-Seed Mean)

| Task | Env Steps | `mjlab` PPO Return (Solved, Wall s) | `SimBaV2 + MuonPPO` Return (Solved, Wall s) |
|---|---:|---:|---:|
| `myoFingerPoseRandom-v0` | `0` | `-90.4` (`26.7%`, `0.0s`) | `-98.0` (`10.0%`, `0.0s`) |
| `myoFingerPoseRandom-v0` | `61,440` | `-30.4` (`46.7%`, `5.7s`) | `68.4` (`65.0%`, `13.3s`) |
| `myoFingerPoseRandom-v0` | `122,880` | `110.0` (`76.7%`, `10.5s`) | `189.2` (`78.3%`, `25.8s`) |
| `myoFingerPoseRandom-v0` | `184,320` | `243.5` (`96.7%`, `15.3s`) | `192.2` (`85.0%`, `38.1s`) |
| `myoFingerReachRandom-v0` | `0` | `-23.1` (`1.7%`, `0.0s`) | `-23.5` (`0.0%`, `0.0s`) |
| `myoFingerReachRandom-v0` | `61,440` | `-3.0` (`0.0%`, `6.5s`) | `19.8` (`6.7%`, `15.2s`) |
| `myoFingerReachRandom-v0` | `122,880` | `94.4` (`18.3%`, `12.9s`) | `104.8` (`25.0%`, `28.5s`) |
| `myoFingerReachRandom-v0` | `184,320` | `234.0` (`33.3%`, `19.0s`) | `165.7` (`35.0%`, `42.5s`) |
| `myoLegStandRandom-v0` | `0` | `659.4` (`93.3%`, `0.0s`) | `660.5` (`95.0%`, `0.0s`) |
| `myoLegStandRandom-v0` | `61,440` | `612.5` (`93.3%`, `10.3s`) | `684.9` (`91.7%`, `17.3s`) |
| `myoLegStandRandom-v0` | `122,880` | `561.2` (`93.3%`, `19.4s`) | `681.6` (`91.7%`, `33.7s`) |
| `myoLegStandRandom-v0` | `184,320` | `541.5` (`91.7%`, `28.6s`) | `676.8` (`91.7%`, `50.1s`) |
| `myoTorsoPoseFixed-v0` | `0` | `-114.3` (`100.0%`, `0.0s`) | `-114.2` (`100.0%`, `0.0s`) |
| `myoTorsoPoseFixed-v0` | `61,440` | `-113.7` (`100.0%`, `6.7s`) | `-114.1` (`100.0%`, `13.8s`) |
| `myoTorsoPoseFixed-v0` | `122,880` | `-109.0` (`100.0%`, `12.5s`) | `-113.5` (`100.0%`, `27.2s`) |
| `myoTorsoPoseFixed-v0` | `184,320` | `-88.4` (`100.0%`, `18.3s`) | `-111.8` (`100.0%`, `40.5s`) |

## Attribution & Licensing

This package is distributed under the Apache License 2.0 and preserves upstream copyright and license notices in [`NOTICE.txt`](NOTICE.txt). It is an adaptation of external research methods — **not** an original MyoSuite learning algorithm — crediting:

- **SimBaV2:** Lee, H., Hwang, D., Kim, D., Kim, H., Tai, J. J., Subramanian, K., Wurman, P. R., Choo, J., Stone, P., & Seno, T. (2025). *Hyperspherical Normalization for Scalable Deep Reinforcement Learning.* [`arXiv:2502.15280`](https://arxiv.org/abs/2502.15280); official code [`DAVIAN-Robotics/SimbaV2`](https://github.com/DAVIAN-Robotics/SimbaV2) (Apache-2.0) and [`xiaohu-art/Weave`](https://github.com/xiaohu-art/Weave).
- **Eden:** Embodied AI Nexus Contributors (2025–2026). *Eden: Generative Simulation and Learning for Embodied AI.* [`embodied-ai-nexus/Eden`](https://github.com/embodied-ai-nexus/Eden) (Apache-2.0).
- **Muon:** Jordan, K., Jin, Y., Boza, V., Jiacheng, Y., Cecista, F., Newhouse, L., & Bernstein, J. (2024). *Muon: An optimizer for hidden layers in neural networks.* [`KellerJordan/Muon`](https://github.com/KellerJordan/Muon) (MIT); PyTorch `torch.optim.Muon` (BSD-3-Clause).
- **HIMLoco & OpenHomie:** Long, J., Wang, Z., Li, Q., Cao, L., Gao, J., & Pang, J. (2024). *Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response.* ICLR 2024, [`arXiv:2312.11460`](https://arxiv.org/abs/2312.11460); [`InternRobotics/HIMLoco`](https://github.com/InternRobotics/HIMLoco) and [`OpenRobotLab/HomieRL`](https://github.com/OpenRobotLab/HomieRL) (MIT).
- **AMP & HUSKY:** Peng, X. B., Ma, Z., Abbeel, P., Levine, S., & Kanazawa, A. (2021). *AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control.* ACM Trans. Graph. (SIGGRAPH 2021), [`arXiv:2104.02180`](https://arxiv.org/abs/2104.02180); [`TeleHuman/humanoid_skateboarding`](https://github.com/TeleHuman/humanoid_skateboarding).
