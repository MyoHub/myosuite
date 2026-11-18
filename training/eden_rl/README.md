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

## Attribution & Licensing

This package is distributed under the Apache License 2.0 and preserves upstream copyright and license notices in [`NOTICE.txt`](NOTICE.txt). It is an adaptation of external research methods — **not** an original MyoSuite learning algorithm — crediting:

- **SimBaV2:** Lee, H., Hwang, D., Kim, D., Kim, H., Tai, J. J., Subramanian, K., Wurman, P. R., Choo, J., Stone, P., & Seno, T. (2025). *Hyperspherical Normalization for Scalable Deep Reinforcement Learning.* [`arXiv:2502.15280`](https://arxiv.org/abs/2502.15280); official code [`DAVIAN-Robotics/SimbaV2`](https://github.com/DAVIAN-Robotics/SimbaV2) (Apache-2.0) and [`xiaohu-art/Weave`](https://github.com/xiaohu-art/Weave).
- **Eden:** Embodied AI Nexus Contributors (2025–2026). *Eden: Generative Simulation and Learning for Embodied AI.* [`embodied-ai-nexus/Eden`](https://github.com/embodied-ai-nexus/Eden) (Apache-2.0).
- **Muon:** Jordan, K., Jin, Y., Boza, V., Jiacheng, Y., Cecista, F., Newhouse, L., & Bernstein, J. (2024). *Muon: An optimizer for hidden layers in neural networks.* [`KellerJordan/Muon`](https://github.com/KellerJordan/Muon) (MIT); PyTorch `torch.optim.Muon` (BSD-3-Clause).
- **HIMLoco & OpenHomie:** Long, J., Wang, Z., Li, Q., Cao, L., Gao, J., & Pang, J. (2024). *Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response.* ICLR 2024, [`arXiv:2312.11460`](https://arxiv.org/abs/2312.11460); [`InternRobotics/HIMLoco`](https://github.com/InternRobotics/HIMLoco) and [`OpenRobotLab/HomieRL`](https://github.com/OpenRobotLab/HomieRL) (MIT).
- **AMP & HUSKY:** Peng, X. B., Ma, Z., Abbeel, P., Levine, S., & Kanazawa, A. (2021). *AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control.* ACM Trans. Graph. (SIGGRAPH 2021), [`arXiv:2104.02180`](https://arxiv.org/abs/2104.02180); [`TeleHuman/humanoid_skateboarding`](https://github.com/TeleHuman/humanoid_skateboarding).
