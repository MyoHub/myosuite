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

## Attribution and licensing

Training methods adapted from or based on external research or code must preserve appropriate attribution.

- Credit the original paper and authors in the method's README, with links to the publication and source repository when available.
- Preserve copyright, license, and attribution notices required by the upstream license.
- Clearly distinguish upstream/adapted code from new MyoSuite-specific implementation.
- Include upstream license files or notices when required.
- Verify that the upstream license permits the intended use and redistribution.
- Do not present an adapted or reproduced method as an original MyoSuite method.

Agents must check attribution and license requirements before adding or modifying externally derived training code.

## Boundary

Keep in `myosuite/`:

- environments and tasks
- observation, reward, and termination contracts
- simulation/backend functionality
- reusable biomechanical/runtime functionality
- lightweight inference/integration interfaces

Keep in `training/`:

- trainers and optimizers
- replay buffers
- world-model training
- experiment configuration
- algorithm-specific evaluation
- other method-specific learning code

When in doubt, keep the training implementation under `training/` and move only clearly reusable runtime functionality into `myosuite/`.

> **MyoSuite defines the simulation and task contract; training methods consume that contract.**
