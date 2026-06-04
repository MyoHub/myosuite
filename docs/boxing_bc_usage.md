# Boxing BC Usage

Two scripts are available:

| Script | Policy format | Use case |
| --- | --- | --- |
| `scripts/boxing_bc.py` | NumPy `.npz` (JAX/Flax weights) | Original export + render |
| `scripts/boxing_bc_torch.py` | PyTorch `.pt` | SB3-compatible inference + fine-tuning |

Use `scripts/boxing_bc.py` for the original JAX-backed policy.
Use `scripts/boxing_bc_torch.py` to export or validate the PyTorch clone.

## Correct boxing environment IDs

Register environments first (e.g. `import myosuite.envs.myo.myochallenge` or
`from myosuite import register_all_envs; register_all_envs()`), then:

| Gymnasium ID | Role |
| --- | --- |
| `myoChallengeBoxingMannequin-v0` | Single-agent vs **scripted mannequin** attacks |
| `myoChallengeBoxingP0-v0` | Single-agent vs **static eight-target** pads (default for long collection / BC) |
| `myoChallengeBoxingVs-v0` | **Two-agent** competitive boxing |

**Removed / not registered:** `myoChallengeBoxingDrill-v0`, `myoChallengeBoxingMannequinMulti-v0`,
`myoChallengeBoxingMannequin-v1` (replaced by `myoChallengeBoxingP0-v0`).

---

## JAX/NumPy Policy (`boxing_bc.py`)

### Export from checkpoint + render

```bash
uv run python scripts/boxing_bc.py \
  --checkpoint Boxing/checkpoint_13114 \
  --motion Boxing/motions/Transitions_mocap/mazen_c3d/punchboxing_push_poses.npz \
  --env-id myoChallengeBoxingP0-v0 \
  --policy-out outputs/debug/mannequin_exact_clone_standalone_clean.npz \
  --video-out outputs/debug/mannequin_exact_clone_standalone_clean_v1.mp4 \
  --report-out outputs/debug/mannequin_exact_clone_standalone_clean_v1_report.json
```

### Run standalone policy (no checkpoint dependency)

```bash
uv run python scripts/boxing_bc.py \
  --policy-in outputs/debug/mannequin_exact_clone_standalone_clean.npz \
  --skip-teacher-compare \
  --motion Boxing/motions/Transitions_mocap/mazen_c3d/punchboxing_push_poses.npz \
  --env-id myoChallengeBoxingP0-v0 \
  --video-out outputs/debug/mannequin_exact_clone_standalone_clean_only_p0.mp4 \
  --report-out outputs/debug/mannequin_exact_clone_standalone_clean_only_p0_report.json
```

---

## PyTorch Policy (`boxing_bc_torch.py`)

### How it works

`scripts/boxing_bc_torch.py` performs a **direct weight transfer** from the
Orbax/Flax checkpoint into a PyTorch `nn.Module` — no BC training required.
The result is numerically identical to the original policy (action MAE < 1e-6).

Architecture: 5 × 1024 hidden layers (Linear → LayerNorm → SiLU) → Linear
output, with obs normalisation baked in as non-learnable buffers.

### Export PyTorch policy + validate

```bash
uv run python scripts/boxing_bc_torch.py \
  --checkpoint Boxing/checkpoint_13114 \
  --motion Boxing/motions/Transitions_mocap/mazen_c3d/punchboxing_push_poses.npz \
  --env-id myoChallengeBoxingP0-v0 \
  --policy-out outputs/bc_torch/boxing_bc.pt \
  --video-out outputs/bc_torch/boxing_bc_torch.mp4 \
  --report-out outputs/bc_torch/boxing_bc_torch_report.json
```

Expected output (300-step rollout):

```json
{
  "steps": 300,
  "parity_mae_random_obs": 4.5e-07,
  "teacher_torch_action_mae_mean": 5.0e-07,
  "teacher_torch_action_mae_p95": 6.8e-07,
  "fell": false
}
```

### Validate from a saved `.pt` (no checkpoint needed)

```bash
uv run python scripts/boxing_bc_torch.py \
  --policy-in outputs/bc_torch/boxing_bc.pt \
  --checkpoint Boxing/checkpoint_13114 \
  --motion Boxing/motions/Transitions_mocap/mazen_c3d/punchboxing_push_poses.npz \
  --env-id myoChallengeBoxingP0-v0 \
  --video-out outputs/bc_torch/boxing_bc_torch_rerun.mp4 \
  --report-out outputs/bc_torch/boxing_bc_torch_rerun_report.json
```

### Use in Python / SB3

```python
import torch
import numpy as np

# Load
model = torch.load("outputs/bc_torch/boxing_bc.pt", map_location="cpu", weights_only=False)
model.eval()

# Inference (SB3-compatible interface)
obs: np.ndarray  # shape (2418,), float32 — raw (unnormalised) observation
action = model.predict(obs)  # shape (354,), float32, values in [-1, 1]

# Convert to muscle activations expected by the boxing env
muscle = np.clip(0.5 * (action + 1.0), 0.0, 1.0)
```

For SB3 training integration, subclass
`stable_baselines3.common.policies.BasePolicy` and set `self.actor = model`.

---

## Notes

- `--checkpoint` + `--motion` are the key inputs for policy export in both scripts.
- Teacher-compare metrics in `boxing_bc.py` are only computed when checkpoint
  loading is enabled (i.e. `--skip-teacher-compare` is not set).
- Experimental scripts and generated mannequin debug artifacts were moved under
  `_untracked/`.
