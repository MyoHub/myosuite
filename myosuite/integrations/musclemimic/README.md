<!--
  Copyright (c) MyoSuite Authors. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# MuscleMimic integration (native MyoSuite)

This package is integration glue for the upstream
[MuscleMimic](https://github.com/amathislab/musclemimic) project, checkpoints,
and retargeted datasets.  Because this code already lives under
`myosuite.integrations.musclemimic`, examples prefer short API names such as
`compile_mimic_fullbody_mjmodel`; cite MuscleMimic when using these assets.

## Citation

If you use the MuscleMimic models, checkpoints, retargeted datasets, policy
loaders, or reproduction tooling from this integration, cite MuscleMimic.  A
BibTeX copy is also shipped at
[`CITATION.bib`](./CITATION.bib).

```bibtex
@article{Li2026MuscleMimic,
  title={Towards Embodied AI with MuscleMimic: Unlocking full-body musculoskeletal motor learning at scale},
  author={Li, Chengkun and Wang, Cheryl and Ziliotto, Bianca and Simos, Merkourios and Kovecses, Jozsef and Durandau, Guillaume and Mathis, Alexander},
  journal={arXiv preprint arXiv:2603.25544},
  year={2026}
}
```

## Motion data, retargeting and licenses

The retargeted motions (`amathislab/musclemimic-retargeted`,
`amathislab/musclemimic-bimanual-retargeted`) are derived from public motion-capture data and were
retargeted to the MyoFullBody and MyoBimanualArm models with GMR.  When you use them, cite the sources too:

| Source | What it provides | Cite |
|---|---|---|
| [AMASS](https://amass.is.tue.mpg.de) | the archive the motions come from (locomotion, transitions, upper limb) | Mahmood et al., ICCV 2019 ([arXiv:1904.03278](https://arxiv.org/abs/1904.03278)) |
| KIT whole-body human motion database | the KIT clips (e.g. `KIT/314/walking_medium09_poses`) inside AMASS | Mandery et al., ICAR 2015; Mandery et al., IEEE T-RO 32(4), 2016 ([doi:10.1109/TRO.2016.2572685](https://doi.org/10.1109/TRO.2016.2572685)) |
| GMR (General Motion Retargeting) | retargeting of the human motion to the musculoskeletal models | Araújo et al., 2025 ([arXiv:2510.02252](https://arxiv.org/abs/2510.02252)); MuscleMimic uses the fork [`gmr_plus`](https://github.com/amathislab/gmr_plus) |
| KINESIS | the KIT training / testing splits (`KIT_KINESIS_*`) | Simos, Chiappa, Mathis, 2025 ([arXiv:2503.14637](https://arxiv.org/abs/2503.14637)) |
| ULTRA-MoCap | the 208 upper-limb trajectories of the bimanual dataset (`ULTRA_MOCAP_*`) | the [ULTRA-MoCap v1 dataset](https://doi.org/10.6084/m9.figshare.28751156.v1) (Fritsche et al.), CC BY 4.0 |

BibTeX entries of these sources are in `myosuite.integrations.citations` (`get_integration_citation("amass")`,
`"kit"`, `"gmr"`, `"kinesis"`). **License:** the datasets derived from AMASS are for non-commercial scientific
research, education or artistic projects under the AMASS license; do not redistribute them. The dataset cards ask you
to cite AMASS and the MuscleMimic paper in resulting publications. The full reference list is the *References* page of
the MyoSuite documentation (`docs/source/references.rst`).

Full-body **trajectory playback**, **preview**, and **MJX smoke** are driven by a
single user-facing entry point:

```bash
uv run myosuite-musclemimic-fullbody-eval [flags...]
```

This integration **does not** read or require anything under the MyoSuite
`sandbox/` directory.

### What the eval CLI does

- **With `--path` and `--motion_path`** (plus `--use_mujoco` and
  `--mujoco_viewer` or `--record`): MyoSuite-native playback — resolves
  the checkpoint reference (`hf://…` or local), loads the motion NPZ from cache
  or disk, steps MuJoCo and shows the viewer. Does **not** import upstream
  `fullbody` / `musclemimic`. If local Orbax policy artifacts are present
  (`train_state` + `config/metadata`), a minimal local policy runner is used;
  otherwise it replays reference trajectory states.
- **Without `--path`** but with `--use_mujoco --mujoco_viewer`:** physics-only
  preview on the MyoFullBody MJCF (random muscle activity). `--path` /
  `--motion_path` are ignored.
- **Without `--path`** and short `--n-steps`:** optional MJX smoke on
  `MjxMimicFullbody-v0` where the MJX extra is installed (see CLI `--help`).

### Demo motion cache (Hugging Face)

**No upstream MuscleMimic package** for cache setup — only `huggingface_hub`:

```bash
pip install huggingface_hub
# or
pip install 'MyoSuite[musclemimic]'
```

Then:

```bash
uv run myosuite-musclemimic-setup-demo-cache
```

Same layout as upstream `musclemimic.utils.demo_cache`.
To relocate the cache, set one of the upstream cache env vars before running
Python scripts:

```bash
export MUSCLEMIMIC_CONVERTED_AMASS_PATH=~/scratch/.musclemimic/caches/AMASS
# or
export CONVERTED_AMASS_PATH=~/scratch/.musclemimic/caches/AMASS
```

## One-time setup

1. **Hugging Face** (gated checkpoints and data):

   ```bash
   uv run hf auth login
   ```

2. **Demo cache** (so `KIT/…` motions resolve):

   ```bash
   uv run myosuite-musclemimic-setup-demo-cache
   ```

   Optional: set `MYOSUITE_MUSCLEMIMIC_ROOT` to a MuscleMimic checkout if you
   prefer that workflow.

## Full snippet: HF trajectory + MuJoCo viewer

From the **MyoSuite repo root** (adjust paths). Install
`MyoSuite[musclemimic]` for `musclemimic_models`.

```bash
uv run hf auth login
uv run myosuite-musclemimic-setup-demo-cache

cd /absolute/path/to/myosuite
uv run myosuite-musclemimic-fullbody-eval \
  --path hf://amathislab/mm-10m-2 \
  --motion_path KIT/314/walking_medium09_poses \
  --use_mujoco \
  --stochastic \
  --eval_seed 0 \
  --n_steps 1000 \
  --mujoco_viewer
```

**Orange reference-site markers** appear automatically over the body — one
semi-transparent sphere per mimic site showing the clip's world-space target
position at each frame. To disable:

```bash
uv run myosuite-musclemimic-fullbody-eval \
  --path hf://amathislab/mm-10m-2 \
  --motion_path KIT/314/walking_medium09_poses \
  --use_mujoco --mujoco_viewer --n_steps 1000 \
  --no_show_targets
```

**Video** (offscreen MP4 export):

```bash
uv run myosuite-musclemimic-fullbody-eval \
  --path hf://amathislab/mm-10m-2 \
  --motion_path KIT/314/walking_medium09_poses \
  --use_mujoco --record --record_path walk_eval.mp4 --record_fps 100
```

**Preview only** (no HF path):

```bash
uv run myosuite-musclemimic-fullbody-eval \
  --use_mujoco --mujoco_viewer --n_steps 1000 --eval_seed 0
```

**MJX smoke** (no `--path`; needs `myosuite[musclemimic,mjx]` on a supported
platform):

```bash
uv run myosuite-musclemimic-fullbody-eval --n-steps 32 --seed 0
# equivalent (explicit):
uv run myosuite-musclemimic-fullbody-eval --backend mjx --n-steps 32 --seed 0
```

**mjlab play** (needs `myosuite[mjlab]`): the mjlab Mimic task is clip-driven, so `--backend mjlab`
requires `--motion_path`. The remaining flags go to `mjlab play` (`--agent zero`, `--num-envs`,
`--viewer`, `--video`, `--checkpoint-file` ...; see `--help` with the same arguments). The
CPU-only flags (`--n_steps`, `--seed`, `--use_mujoco` ...) are consumed and ignored.

```bash
uv run myosuite-musclemimic-fullbody-eval --backend mjlab \
  --motion_path KIT/314/walking_medium09_poses --agent zero --num-envs 1
```

> **Note:** mjlab is a GPU-parallel training backend (MuJoCo Warp / Isaac Lab).
> It runs hundreds of environments in parallel on the GPU and does **not**
> provide an interactive MuJoCo viewer.  There is no `--backend mjlab` equivalent
> of `--use_mujoco --mujoco_viewer` — use the CPU/MuJoCo path for visual
> inspection and mjlab for large-scale RL training.  See the section below for
> Python-level mjlab training usage.

## mjlab backend (GPU parallel training)

mjlab (`MuJoCo Warp / Isaac Lab`) runs `myoMimicFullbody-v0` with thousands of
parallel environments on GPU.  It is a **training** backend — there is no
interactive viewer equivalent to `--use_mujoco --mujoco_viewer`.

### Visualization (mjlab native viewer)

mjlab's `NativeMujocoViewer` syncs GPU simulation state to CPU each frame and
displays it in MuJoCo's passive viewer:

```
GPU step → qpos[env_idx].cpu().numpy() → cpu_data.qpos[:] → mj_forward → viewer.sync()
```

```bash
# Zero-action run — verify the env builds and the viewer opens (needs a display; --motion_path is required)
uv run myosuite-musclemimic-fullbody-eval --backend mjlab \
  --motion_path KIT/314/walking_medium09_poses --agent zero

# Random policy
uv run myosuite-musclemimic-fullbody-eval --backend mjlab \
  --motion_path KIT/314/walking_medium09_poses --agent random

# Trained checkpoint (local file)
uv run myosuite-musclemimic-fullbody-eval --backend mjlab \
  --motion_path KIT/314/walking_medium09_poses --checkpoint path/to/ckpt.pt

# Trained checkpoint from Weights & Biases
uv run myosuite-musclemimic-fullbody-eval --backend mjlab \
  --motion_path KIT/314/walking_medium09_poses --wandb-run-path org/project/run-id
```

Or call mjlab's `play` CLI directly (task must be registered first):

```bash
uv run play myoMimicFullbody-v0 --agent zero
uv run play myoMimicFullbody-v0 --wandb-run-path org/project/run-id
```

### ONNX policy on mjlab GPU env

mm-10m-2 was trained on `musclemimic_models` 1.0.5: run it with
`MYOSUITE_MUSCLEMIMIC_MODELS_VERSION=1.0.5` (see [Model release](#model-release)).

mjlab's `myoMimicFullbody-v0` env produces ~684-dim observations — too small
for the mm-10m-2 model which expects 2418 dims.  `FullbodyOnnxMjlabPolicy`
(and `FullbodyOrbaxMjlabPolicy`) rebuild the checkpoint observation after each
step from mjlab's batched sim data, on its device:

```
GPU step → TorchFullbodyObsAdapter (all selected envs, 2418 dims, float32)
         → onnxruntime (one host copy of the batch) → actions → GPU
```

There is no CPU `mj_forward` and no per-env copy.  The observation is float32
and matches `FullbodyObsAdapter` to float32 rounding on the same sim arrays;
compared with a CPU `mj_forward` of the same state, muscles on a few wrapped
tendons and the foot touch sensors can differ (MuJoCo Warp computes them in
float32 with its own contact solver).  `obs_backend="cpu"` keeps the previous
path (per-env GPU→CPU copy, `mj_forward`, `FullbodyObsAdapter`) for debugging
and parity checks; pass `env_indices=range(env.num_envs)` to get one action per
env instead of broadcasting env 0's.

The CLI's `--onnx` option runs the CPU MuJoCo loop instead (no warp):

```bash
uv run myosuite-musclemimic-fullbody-eval \
  --backend mjlab \
  --path hf://amathislab/mm-10m-2 \
  --motion_path KIT/314/walking_medium09_poses \
  --onnx mm-10m-2.onnx
```

`--path` is optional but recommended — it provides the `goal_params` metadata
needed to correctly initialise `FullbodyObsAdapter` (lookahead steps, site
list, etc.).  Without it, adapter defaults are used.

**Programmatic usage:**

```python
import mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer import NativeMujocoViewer

from myosuite.core.trajectory_io import load_motion_clip, resolve_motion_path
from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import register_mimic_mjlab_tasks_with_clip
from mjlab.tasks.registry import register_mjlab_task
from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import _elbow_ppo_runner_cfg
from myosuite.integrations.musclemimic.fullbody_local_policy import FullbodyObsAdapter
from myosuite.integrations.musclemimic.fullbody_model import (
    compile_mimic_fullbody_mjmodel,
    default_mimic_fullbody_config,
)
from myosuite.integrations.musclemimic.mjlab_onnx_policy import FullbodyOnnxMjlabPolicy

# Load clip and register trajectory-mode env
motion_file = resolve_motion_path("KIT/314/walking_medium09_poses")
clip = load_motion_clip(motion_file, expected_nq=89, expected_nv=88)
register_mimic_mjlab_tasks_with_clip(register_mjlab_task, _elbow_ppo_runner_cfg, clip=clip)

# CPU model for obs adapter
cpu_model, _, _ = compile_mimic_fullbody_mjmodel(default_mimic_fullbody_config())
obs_adapter = FullbodyObsAdapter(model=cpu_model, clip=clip, goal_params={})

# Build mjlab GPU env
env = ManagerBasedRlEnv(cfg=load_env_cfg("myoMimicFullbody-v0", play=True), device="cpu")
env_wrapped = RslRlVecEnvWrapper(env, clip_actions=True)

# ONNX policy (obs normalization baked in; no orbax required)
policy = FullbodyOnnxMjlabPolicy(
    env=env_wrapped,
    cpu_model=cpu_model,
    obs_adapter=obs_adapter,
    onnx_path="mm-10m-2.onnx",
    clip=clip,
)

NativeMujocoViewer(env_wrapped, policy).run()
env.close()
```

### Python training usage

```python
import torch
import mjlab.envs as mjlab_envs
from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import register_mjlab_tasks

# Register all MyoSuite tasks on mjlab (do once at startup)
register_mjlab_tasks()

# Build parallel env (default: 1024 envs on GPU)
env = mjlab_envs.make("myoMimicFullbody-v0")
obs, _ = env.reset()

for _ in range(1000):
    # env.num_actions and env.device are set by mjlab ManagerBasedRlEnv
    action = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

### Trajectory-mode training (motion clip as target source)

```python
from myosuite.core.trajectory_io import load_motion_clip, resolve_motion_path
from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import register_mimic_mjlab_tasks_with_clip
from mjlab.tasks.registry import register_mjlab_task
from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import _elbow_ppo_runner_cfg
import mjlab.envs as mjlab_envs

motion_file = resolve_motion_path("KIT/314/walking_medium09_poses")
clip = load_motion_clip(motion_file, expected_nq=89, expected_nv=88)

# Register with clip: targets follow the motion clip instead of random box
register_mimic_mjlab_tasks_with_clip(
    register_mjlab_task, _elbow_ppo_runner_cfg, clip=clip
)
env = mjlab_envs.make("myoMimicFullbody-v0")
```

In trajectory mode, three additional observation terms are available:
`clip_ref_qpos`, `clip_ref_qvel`, `clip_phase`.

Motion NPZ files may also store partial `qpos` / `qvel` arrays when they
include `qpos_names` / `qvel_names` or shared `joint_names` metadata. The
shared loader accepts those clips, and the CPU baseline / mjlab mimic paths
expand them against the target MuJoCo model so qpos/qvel tracking applies only
on that named joint subset.

### Requirements

```bash
pip install 'myosuite[mjlab]'
# or
uv sync --extra mjlab
```

GPU (CUDA) recommended. See mjlab documentation for full hardware requirements.

## Target-site visualization (viewer markers)

When `--use_mujoco --mujoco_viewer` is active, a small orange sphere is drawn
at each reference site position from `clip.site_xpos` on every rendered frame.
This lets you visually compare where the body segments are versus where the
reference motion places them.

| Flag | Effect |
|------|--------|
| *(default)* | Markers on — one sphere per mimic site |
| `--no_show_targets` | Markers off |

The markers are rendered via MuJoCo's `viewer.user_scn` (no MJCF changes) and
are available in both trajectory-replay mode and local-policy-inference mode.

To use the same machinery programmatically:

```python
from myosuite.core.site_marker_viz import make_site_marker_viz_fn
from myosuite.core.mujoco_playback import run_passive_viewer_loop

viz_fn = make_site_marker_viz_fn(
    clip=clip,
    site_names=tuple(site_names),
    model=model,
    rgba=(1.0, 0.4, 0.0, 0.6),  # orange, 60 % opacity
    sphere_radius=0.03,
)

run_passive_viewer_loop(
    model=model,
    data=data,
    n_steps=n_steps,
    step_fn=my_step_fn,
    viz_fn=viz_fn,
)
```

## ONNX export

Export the full-body policy to a single self-contained ONNX file for
cross-backend playback, deployment, or regression testing.

```bash
# Requires: orbax-checkpoint, torch (CPU-only is sufficient)
uv run myosuite-export-onnx export \
  --framework orbax \
  --checkpoint hf://amathislab/mm-10m-2 \
  --output mm-10m-2.onnx
```

This produces a single `mm-10m-2.onnx` file (~38 MB) with the following interface:

| | Name | Shape | Dtype |
|---|---|---|---|
| Input | `obs` | `(batch, 2418)` | float32 |
| Output | `action` | `(batch, 354)` | float32 |

**Obs normalization** is baked into the ONNX graph — pass raw (unnormalized)
observations directly.  Actions are clipped to `[-1, 1]`.

### Inference with onnxruntime

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("mm-10m-2.onnx", providers=["CPUExecutionProvider"])

obs = np.zeros((1, 2418), dtype=np.float32)   # your observation vector
action = sess.run(["action"], {"obs": obs})[0]  # shape (1, 354)
```

### Using the ONNX policy for playback and recording

The `--onnx` flag wires the ONNX policy into the native playback runner.
Obs normalization is baked into the ONNX graph — no Orbax/JAX required at
inference time.

**Interactive viewer (macOS requires `mjpython`):**

```bash
uv run myosuite-musclemimic-fullbody-eval \
  --path hf://amathislab/mm-10m-2 \
  --motion_path KIT/314/walking_medium09_poses \
  --use_mujoco --mujoco_viewer \
  --n_steps 1000 --eval_seed 0 \
  --onnx mm-10m-2.onnx
```

**Video recording (headless, works everywhere):**

```bash
uv run myosuite-musclemimic-fullbody-eval \
  --path hf://amathislab/mm-10m-2 \
  --motion_path KIT/314/walking_medium09_poses \
  --use_mujoco --record --record_path onnx_walk.mp4 \
  --n_steps 1000 --eval_seed 0 \
  --onnx mm-10m-2.onnx
```

**Programmatic usage:**

```python
import mujoco
import numpy as np
from myosuite.integrations.musclemimic.fullbody_model import (
    compile_mimic_fullbody_mjmodel,
    default_mimic_fullbody_config,
)
from myosuite.integrations.musclemimic.fullbody_local_policy import (
    FullbodyObsAdapter,
    OnnxPolicyRunner,
)
from myosuite.core.trajectory_io import load_motion_clip, resolve_motion_path

model, _, _ = compile_mimic_fullbody_mjmodel(default_mimic_fullbody_config())
data = mujoco.MjData(model)
motion_file = resolve_motion_path("KIT/314/walking_medium09_poses")
clip = load_motion_clip(motion_file, expected_nq=model.nq, expected_nv=model.nv)

# goal_params from the checkpoint's config/metadata (or {} for defaults)
obs_adapter = FullbodyObsAdapter(model=model, clip=clip, goal_params={})
runner = OnnxPolicyRunner(
    onnx_path="mm-10m-2.onnx",
    obs_dim=2418,
    action_dim=354,
    obs_adapter=obs_adapter,
)

data.qpos[:] = clip.qpos[0]
mujoco.mj_forward(model, data)
for i in range(len(clip.qpos)):
    action = runner.action_for(data, clip, i)
    runner.step(model, data, action)
```

> **Why not use the ONNX model directly with mjlab's env?**
> mjlab's `myoMimicFullbody-v0` env produces ~684-dim observations
> (qpos + qvel + act + site positions), while the mm-10m-2 ONNX model expects
> 2418-dim observations built by `FullbodyObsAdapter`. Using the ONNX model
> requires the CPU-based playback path with `FullbodyObsAdapter`.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `MYOSUITE_PREVIEW_REEXEC_MJPYTHON` | Internal guard for one-time preview relaunch under `mjpython` on macOS |
| `MYOSUITE_MUSCLEMIMIC_DISABLE_LOCAL_POLICY` | Set to `1` to force trajectory replay even when local Orbax artifacts are present |
| `MYOSUITE_MUSCLEMIMIC_USE_UPSTREAM` | Set to `1` to delegate `--path --mujoco_viewer` to upstream `fullbody.eval` instead of the native runner |
| `MYOSUITE_MUSCLEMIMIC_MODELS_VERSION` | `musclemimic_models` release the models are built as when the config sets no `model_version` (default `1.0.6`; `1.0.5` for checkpoints trained on it) |

### Model release

MyoSuite installs `musclemimic_models==1.0.6`, which fixed the left knee (two coupling
polynomials were negated, so at 1 rad of flexion the left tibia rotated -0.25 rad about its
axis where the right one rotated +0.25 rad) and six muscle wraps. `amathislab/mm-10m-2` was
trained on 1.0.5 and falls within 4 s on 1.0.6. `model_version="1.0.5"` in the model config
rebuilds 1.0.5 bit-exactly from the installed package;
`model_versions.checkpoint_models_version(ref)` returns the release of a published checkpoint,
and the tutorials and the playback / eval CLIs use it. For envs that build their config
internally (e.g. the mjlab Mimic env), set `MYOSUITE_MUSCLEMIMIC_MODELS_VERSION=1.0.5`.

## Troubleshooting

**`failed to dlopen` / `libpython` (macOS + `mjpython`)**

Do not start MyoSuite as the outer process with bare `mjpython` from a **uv**
venv if the linker cannot resolve `libpython`. Prefer:

```bash
uv run myosuite-musclemimic-fullbody-eval …
```

Preview mode may still re-exec under `mjpython` when it is on `PATH`; if that
fails, use a non-uv Python (conda/Homebrew venv) per MuJoCo docs.

**`MJX stack not available`**

MJX smoke needs a compatible MuJoCo + `mujoco.mjx` stack (often **Linux** with
`pip install 'MyoSuite[musclemimic,mjx]'`).

**`--path` run fails (“Native playback failed”)**

Requires valid `--path`, resolvable `--motion_path`, and
`--use_mujoco` with `--mujoco_viewer` (or recording flags as implemented).

---

Programmatic Mimic envs (`myoMimic*`, MJX, mjlab) live in the main package and
tests; this README intentionally documents only the **`myosuite-musclemimic-fullbody-eval`**
CLI for end-user playback and smoke.
