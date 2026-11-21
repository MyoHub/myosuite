# `mm-evalkit` & Musculoskeletal EMG Validation Boundary

Musculoskeletal validation across MyoHub is split into three clean layers so neither `myo_sim`, `myosuite`, nor [`amathislab/mm-evalkit`](https://github.com/amathislab/mm-evalkit) duplicates another package's responsibility:

| Layer | Repository / Package | Responsibility |
|---|---|---|
| **1. Structural Model Gates** | `MyoHub/myo_sim` (`myo_sim#197`) | Fast (`< 3 s`), dataset-free structural checks on compiled XMLs: muscle operating ranges, moment-arm sign/coupling invariants, passive-force bounds, dead actuators, and parameter literature provenance. |
| **2. Physics & Inverse Solvers** | `myosuite.biomechanics` (`NumPy`/`SciPy`/`MuJoCo`/`OSQP` only) | Pure numerical solvers turning kinematics and external loads into predicted muscle excitations/activations (`inverse_dynamics`, `static_optimization`, `computed_muscle_control`, `emg`, `emg_validation`). Carries **zero** dataset I/O, YAML, or external evaluation dependencies. |
| **3. Datasets, Stats & Rollout Evaluation** | `amathislab/mm-evalkit` + `myosuite.integrations.{emg,mm_evalkit}` | Dataset adapters (`mm_evalkit.datasets` entry points), paired cohort statistics (`mm_evalkit.stats.rsa`, `mm_evalkit.stats.wilcoxon`), and standardized rollout `.npz` / `policies.yaml` evaluation (`mm_evalkit.workflows`). |

## Wheel vs Source-Checkout Capabilities

Both `mm-evalkit evaluate <dataset>` (from an installed `myosuite[eval]` wheel) and `python scripts/validation/validate_emg.py` (from a source checkout) invoke the **same** workflow in `myosuite.integrations.mm_evalkit.workflow`:

- **Available from wheel and source checkout:**
  - All registered `mm_evalkit.datasets` adapters (`rajagopal2016`, `lencioni2019`, `dimitrov2023`, `camargo2021`, `custom-emg`, plus upper-limb/hand sources when installed).
  - Target models specified as `.xml`/`.mjb` files, `myo_sim:<model_name>`, installed `myo_sim` model names, `myoMimic` body names, or registered `make_env(<env_id>)` IDs.
- **Source-checkout only (`CheckoutCapabilityError`):**
  - `<worktree_dir>:<model_name>` compiles an unmerged local `myo_sim` git worktree in an isolated subprocess and requires `<worktree_dir>/myo_sim/` to exist on disk.
