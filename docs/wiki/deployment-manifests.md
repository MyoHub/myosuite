# Deployment manifests

Record the **actual configured model**, rather than a fresh default env, when scaling, strength, mass, wrappers or
timing differ from the registration. `manifest.json` schema 2 keeps the schema-1 contract fields and can add a
versioned `deployment` section. Old manifests remain readable. Explicit `contract=...` calls remain metadata-only;
pass `env=actual_env` or `deployment=...` to attach physical provenance.

Backfilling a manifest describes today's deployment target; weights alone cannot prove which physical model was
used during training. For reproducibility, write the manifest when saving the checkpoint from the actual env and
retain that original record instead of regenerating it against a later model release.

```python
from myosuite import make_env
from myosuite.utils.checkpoint_manifest import write_manifest

with make_env("myoElbowPose1D6MFixed-v0") as env:
    write_manifest("checkpoints/elbow", "myoElbowPose1D6MFixed-v0", env=env)
```

The deployment section contains:

| Field | Meaning |
|---|---|
| `model.physical_sha256` | Exact compiled model arrays, options, names and embedded assets, excluding machine-local source paths |
| `model.kinematic_fingerprint` | Existing rounded body/joint/coupling compatibility hash, with no landmark/sole targets; cannot establish physical identity |
| `artifacts` | Bundle-relative file paths, byte SHA-256 and sizes; no model weights are embedded in JSON |
| `checkpoint` | Source checkpoint byte identity; provenance only, need not ship with ONNX |
| `references` | Named motion/calibration input hashes; pinned HF provenance can accompany each record |
| `interface` | Ordered observations/actions, shapes, units, history and action/command semantics supplied by the caller |
| `normalization` | Frozen normalization and whether it is embedded; absent details remain `unknown` |
| `timing` | Physics timestep, control timestep and exact integer decimation |
| `physics_options` | Native solver, contact and other MuJoCo option values |
| `runtime` | Backend, build-time Python/platform/packages plus explicit target runtime versions, without importing optional training libraries |

The physical hash algorithm is versioned and tied to the recorded MuJoCo version. Native versions may produce
different compiled representations for equivalent XML. Recheck physics/parity before overriding a difference.
It includes nominal model values; it does not describe a domain-randomization distribution, arbitrary Python code
or mutable episode state. Record wrapper/command/randomization configuration in the explicit interface metadata.
For source-byte provenance, also hash the source XML and assets; compiled identity cannot recover discarded comments
or the original file format. These hashes detect differences; they are not signatures or evidence of policy quality.

For a browser or custom deployment, use the compiled deployment model and explicit metadata:

```python
from myosuite.utils.deployment_manifest import (
    artifact_record, build_deployment_manifest,
    validate_deployment_manifest, write_deployment_manifest,
)

manifest = build_deployment_manifest(
    model,
    interface={
        "observations": [{"name": "joint_angle", "shape": [1], "units": "rad"}],
        "actions": [{"name": "muscle", "shape": [6], "units": "excitation"}],
        "history": {"length": 1, "order": "term-major"},
        "action_mapping": "sigmoid before actuator write",
    },
    control_dt=0.02,
    artifacts={"policy.onnx": "export/policy.onnx"},
    checkpoint=artifact_record("checkpoints/model_200.pt"),
    references={"motion": artifact_record("clips/walk.npz", source={
        "repo_id": "owner/repository", "filename": "clips/walk.npz",
        "revision": "<pinned tag or full commit>",
    })},
    normalization={"kind": "embedded", "input": "raw"},
    backend="browser",
    runtime={"mjswan": "<tested version>", "mujoco_wasm": "<tested version>"},
)
write_deployment_manifest("export/policy.onnx", manifest)
# At playback construct current from the actual runtime configuration/references:
validate_deployment_manifest(manifest, current=current, artifact_root="export")
```

Observation/action lists are ordered: changing their order must fail even if the width stays the same. The CPU
env helper records spaces, ordered obs keys and actuator names automatically. Units, history and action mapping
remain null because a generic env cannot safely infer them. Supply complete metadata before treating a manifest as
a reproducible deployment. `artifact_record` rejects default HF branches; use a pinned tag/commit and, when known,
`resolved_commit`. It performs no download. Keep checkpoints on Hugging Face, never commit them to git.

The ONNX CLI can attach a sidecar without changing existing export calls:

```bash
python -m myosuite.utils.export_onnx export --framework rslrl \
  --checkpoint model_200.pt --obs-dim 9 --act-dim 6 --output export/policy.onnx \
  --env myoElbowPose1D6MFixed-v0
python -m myosuite.utils.export_onnx verify --onnx export/policy.onnx \
  --env myoElbowPose1D6MFixed-v0
```

Use `--deployment-metadata metadata.json` with `interface`, `normalization`, `references` and optional
`checkpoint_source` and `runtime` fields to record details the env cannot infer. Reference entries are `artifact_record` outputs.
For HF checkpoints download at a pinned revision first and export the local path; provenance mode rejects `hf://`
resolution through an unpinned default. Directory checkpoints get a sorted per-file content digest.

`load_policy` and ONNX CPU verification validate new manifests before deserializing/inferencing. Missing legacy
manifests retain their previous behavior. Automatic validation authenticates frozen normalization through checkpoint
bytes; custom references/I/O require `current_deployment` built from the real configuration. Runtime backend and
optional dependency differences are recorded but are not themselves incompatibilities: CPU, GPU and browser differ.
The exact model hash still checks native MuJoCo version. Override only reviewed differences with
`allow_provenance_mismatch=True` (`verify --allow-provenance-mismatch`); every difference produces a warning.
