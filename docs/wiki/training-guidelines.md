# Training Guidelines

> Related proposal: #580

MyoSuite has three layers: tasks (registered env ids), models (myo_sim) and learning methods. Methods live in the installable package under `myosuite/methods/<method>/` (for example `myosuite/methods/dynsyn/`).

## Placement

```text
myosuite/methods/<method>/     # the method package, with its upstream LICENSE / NOTICE files next to the code
scripts/<method>/              # command lines (not shipped)
myosuite/tests/test_<method>_*.py
docs/wiki/<method>.md          # usage, recipe, results
```

A method takes a task id, a model and a control step and builds its env through `make_env(EnvConfig(env_id, ctrl_dt=...), backend=...)`. Tasks never import methods.

## Dependencies

Methods add no separate distribution and no new extra. GPU methods use the existing `[mjlab]` extra (mjlab brings torch, tensordict and tyro); raise its floors only when a method needs it.

- `import myosuite` and `import myosuite.methods` must not import torch or mjlab.
- A method package that lacks its dependencies raises an `ImportError` naming `pip install 'myosuite[mjlab]'`.
- Tests skip cleanly without them (`pytest.importorskip("torch")`, `pytest.importorskip("mjlab")`).

`scripts/train_mjlab.py`, `scripts/train_sb3.py`, and the root `rl` and `mjlab` extras stay where they are. Older projects under `training/` predate this layout.

## Method options

Some methods are options of a trainer rather than trainers of their own. They change how a policy acts on a task, not the task or the model, so they stay opt-in and live under `myosuite/methods/` too.

- **DynSyn** (`myosuite/methods/dynsyn/`, He et al., ICML 2024): one action per muscle group, optionally plus bounded per-muscle corrections. Supported by PPO: `scripts/dynsyn_groups.py` computes the groups, `scripts/train_mjlab.py --dynsyn-groups <groups.json>` trains with them, and every checkpoint stores the setup, so `scripts/eval_mjlab_policy.py` and `--agent.resume` reuse it. CPU envs take `DynSynWrapper`. Other methods (e.g. SuperMimic) could adopt it later through the same mjlab action term.

## Attribution and licensing

Training methods adapted from or based on external research or code must preserve appropriate attribution.

- Credit the original paper and authors in the method's wiki page and module docstring, with links to the publication and source repository when available.
- Preserve copyright, license, and attribution notices required by the upstream license.
- Clearly distinguish upstream/adapted code from new MyoSuite-specific implementation.
- Include upstream license files or notices when required.
- Verify that the upstream license permits the intended use and redistribution.
- Do not present an adapted or reproduced method as an original MyoSuite method.

Agents must check attribution and license requirements before adding or modifying externally derived training code.

## Boundary

Keep in the task and core layers of `myosuite/`:

- environments and tasks
- default per-task runner configurations registered with their environments through `register_cpu_twins` / `register_mjlab_task` (for example, `myo_ppo_runner_cfg` in `myosuite/envs/myo/backends/mjlab/tasks/rl.py`)
- observation, reward, and termination contracts
- simulation/backend functionality
- reusable biomechanical/runtime functionality
- lightweight inference/integration interfaces

Keep in `myosuite/methods/<method>/`:

- trainers and optimizers
- replay buffers
- world-model training
- method-specific experiment configuration and overrides beyond the registered task defaults
- algorithm-specific evaluation
- other method-specific learning code

When in doubt, keep code in the method and move only clearly reusable runtime functionality into the task or core layers.

> **MyoSuite defines the simulation and task contract; training methods consume that contract.**
