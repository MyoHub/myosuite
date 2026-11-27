# Developer Guide

Technical reference for building tasks, observation/reward/action terms, GPU backends, deployment bundles, and external training packages in MyoSuite.

> **Source-of-truth priority:** Code and tests → [AGENTS.md](https://github.com/MyoHub/myosuite/blob/dev/AGENTS.md) → this guide.

---

## 1. Onboarding, Architecture & Standards

| Page | Scope |
| :--- | :--- |
| [Getting Started](getting-started.md) | **New to the codebase — start here:** local environment setup, test suite, and contribution loop |
| [Repository Map](repository-map.md) | Package layout, subpackage boundaries, and where each class of change belongs |
| [Engineering Standards](engineering-standards.md) | Architectural invariants, quality gates, parity baselines, and new-environment checklist |
| [Library Usage](library-usage.md) | Approved runtime primitives and helpers — check before writing any wrapper or utility |
| [Canonical Model Scaling](scaling-migration.md) | Subject geometry and mass scaling facade over `myo_sim.scaling` (`make_env(..., scales=...)`) |

---

## 2. Environments, Terms & Cross-Backend Deployment

| Page | Scope |
| :--- | :--- |
| [Adding a New Task](adding-a-new-task.md) | Step-by-step worked example for registering a new task across CPU and GPU backends |
| [Writing Term Functions](writing-term-functions.md) | Authoring composable observation, reward, action, and termination term functions |
| [mjlab Design Guide](mjlab-design-guide.md) | Designing and verifying `mjlab` (`mujoco-warp`) GPU environment twins |
| [Cross-Backend Contract](cross-backend-contract.md) | Observation, action, reset, and ONNX parity contract across CPU, `mjlab`, MJX, and browser runtimes |
| [Deployment Manifests](deployment-manifests.md) | Recording physical model SHA-256, ordered I/O contracts, and pinned checkpoint provenance (`manifest.json` v2) |
| [BiomechPriorVAE Proposal](biomech-prior-integration.md) | Proposed offline gait-prior evaluation and optional CPU/mjlab reward |

---

## 3. Training, Evaluation & Contributor Workflow

| Page | Scope |
| :--- | :--- |
| [Training Guidelines](training-guidelines.md) | Isolating `training/<method>/` packages, dependency boundaries, licensing, and the 4-task PPO benchmark |
| [myoSuperMimic Guide](https://github.com/MyoHub/myosuite/blob/dev/training/supermimic/README.md) | Training, evaluating, and fine-tuning full-body imitation policies with `myoSuperMimic` |
| [Performance Cards](performance-cards.md) | Reporting timing, memory, and multi-seed evaluation metrics in reproducible performance cards |
| [Paired Push Evaluation](paired-push-evaluation.md) | Deterministic and velocity-jittered paired perturbation recovery protocol (`scripts/eval_push_recovery.py`) |
| [Agent Workflow](agent-workflow.md) | Step-by-step contributor and coding-agent workflow |
| [Maintenance Log](log.md) | Chronological log of architectural additions and contract changes on `dev` |
