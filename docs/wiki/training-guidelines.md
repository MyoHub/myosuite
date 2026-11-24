# Training Guidelines

> Related proposal: #580

MyoSuite provides environments, simulation, task definitions, and reusable runtime functionality. Training algorithms and experimental learning methods should stay outside the installable `myosuite` package.

## Placement

Add method-specific training code under top-level:

```text
training/<method>/
```

Each method should be a self-contained Python project with its own `pyproject.toml`, dependencies, tests, README, and optional lockfile.

Do not add method-specific training dependencies to the root MyoSuite `pyproject.toml`.

## Dependency isolation

Different methods may require incompatible versions of PyTorch, JAX, mjlab, or other research dependencies. Do not assume training methods can share one environment.

Each method should be independently installable, for example:

```bash
pip install myosuite
pip install "git+https://github.com/MyoHub/myosuite.git@dev#subdirectory=training/<method>"
```

Use separate virtual environments when dependency stacks conflict.

## Existing training entry points and extras

`scripts/train_mjlab.py`, `scripts/train_sb3.py`, and the root `rl` and `mjlab` extras stay in their current locations. This guideline does not require moving or removing them. Any future migration should be proposed and reviewed separately, preserving supported training commands and dependency installation paths. New method-specific trainers and dependencies follow the `training/<method>/` layout above.

## Attribution and licensing

Training methods adapted from or based on external research or code must preserve appropriate attribution.

- Credit the original paper and authors in the method's README, with links to the publication and source repository when available.
- Preserve copyright, license, and attribution notices required by the upstream license.
- Clearly distinguish upstream/adapted code from new MyoSuite-specific implementation.
- Include upstream license files or notices when required.
- Verify that the upstream license permits the intended use and redistribution.
- Do not present an adapted or reproduced method as an original MyoSuite method.

Agents must check attribution and license requirements before adding or modifying externally derived training code.

## Benchmark against PPO

Every new method under `training/<method>/` includes a benchmark against PPO on **all four tasks** below. CPU training and evaluation are the minimum requirement; benchmarking both CPU and GPU (mjlab) is preferred. GPU-only results or CPU playback of a GPU-trained policy do not satisfy the CPU requirement.

| Task | Registered environment | Definition |
|---|---|---|
| Elbow random reaching (random target joint angle) | `myoElbowPose1D6MRandom-v0` | [Basic task registration](../../myosuite/envs/myo/tasks/basic/__init__.py) |
| Single-arm MyoChallenge pick and place | `myoChallengeRelocateP1-v0` | [Challenge task registration](../../myosuite/envs/myo/tasks/challenge/__init__.py) |
| Fullbody random directional locomotion | `myoFullBodyDirectional-v0` | [Directional task](../../myosuite/envs/myo/tasks/mimic/cpu.py) |
| Fullbody motion mimic | `myoMimic-v0` (default `myofullbody` model and `supermimic` preset) | [Mimic task registration](../../myosuite/envs/myo/tasks/mimic/__init__.py) |

Use the registered defaults, including randomized targets/headings and the motion distribution. `myoMimicFullbody-v0` is random-target site tracking, not a substitute for motion mimic. Any task or backend exception requires explicit human approval linked in the PR and an explanation in the method's README; missing results must not be presented as benchmark completion.

Run the method's own trainer and PPO side by side on each backend being reported: CPU PPO through `scripts/train_sb3.py --algo ppo`, and GPU PPO through `scripts/train_mjlab.py` with the registered default runner (`myo_ppo_runner_cfg`). Compare within each backend and report CPU and GPU results separately:

- Use the four registered `env_id`s above, with unmodified observations, actions and rewards. Report any task-side change separately.
- Use the same seeds, at least three per arm, and the same budget in environment steps. Report wall-clock time and hardware too.
- Report final success rate or return as mean ± std over seeds, learning curves against environment steps and against wall-clock time, and peak memory or GPU hours.
- State in one sentence how the method differs, with the numbers that show it: for example better final performance, more sample-efficient, faster in wall-clock, more robust, or able to do something PPO cannot.

Put the table and curves in the method's README. Pin the checkpoints and evaluation logs on Hugging Face at a revision (see [AGENTS.md](../../AGENTS.md), *Model weights*). For methods where PPO is not the natural reference (imitation or offline learning, for example), add the closest standard baseline, such as behaviour cloning, and keep PPO as the common reference row.

## Boundary

Keep in `myosuite/`:

- environments and tasks
- default per-task runner configurations registered with their environments through `register_cpu_twins` / `register_mjlab_task` (for example, `myo_ppo_runner_cfg` in `myosuite/envs/myo/backends/mjlab/tasks/rl.py`)
- observation, reward, and termination contracts
- simulation/backend functionality
- reusable biomechanical/runtime functionality
- lightweight inference/integration interfaces

Keep in `training/`:

- trainers and optimizers
- replay buffers
- world-model training
- method-specific experiment configuration and overrides beyond the registered task defaults
- algorithm-specific evaluation
- other method-specific learning code

When in doubt, keep the training implementation under `training/` and move only clearly reusable runtime functionality into `myosuite/`.

A loader for one method's checkpoints is not a reason to copy that method's network into `myosuite/`. Have the method export a format MyoSuite already loads (ONNX through `myosuite.utils.export_onnx`, or a plain `nn.Module` / TorchScript artifact). If a custom architecture must be rebuilt, the training package registers it; the architecture code stays in `training/<method>/`.

Reviewers check this boundary on every PR that touches `myosuite/`: architectures, trainers, optimizer settings beyond registered defaults, replay buffers and self-play pools in the installable package are a placement violation, even when the PR's tests pass.

> **MyoSuite defines the simulation and task contract; training methods consume that contract.**
