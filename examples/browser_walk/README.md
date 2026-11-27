# Directional walking in the browser

This pilot exports the existing MyoSuite directional-forward checkpoint on its own
MyoLeg model. It uses mjswan's viewer, MuJoCo WASM, ONNX Runtime Web and traced MDP
terms. Weights come from pinned Hugging Face commit
`ecfd7ed0d8b2b3211fe73b14d9538b419970d6df`; no weights or anatomy assets enter git.

Use a separate Python 3.11–3.13 build environment:

```powershell
uv venv --python 3.12 tasks/browser-env
uv pip install --python tasks/browser-env/Scripts/python.exe -e ".[mjlab]" -r examples/browser_walk/requirements.txt
$env:PYTHONUTF8 = "1"
tasks/browser-env/Scripts/python.exe scripts/build_browser_walk.py --output tasks/browser-walk --serve
```

The pilot depends on [deployment manifests #724](https://github.com/MyoHub/myosuite/pull/724)
and the merged [entity-state writes #728](https://github.com/MyoHub/myosuite/pull/728),
plus mjswan [#216](https://github.com/ttktjmt/mjswan/pull/216),
[#217](https://github.com/ttktjmt/mjswan/pull/217) and
[#218](https://github.com/ttktjmt/mjswan/pull/218) for static layout reads, class resets
and valid constant graphs. Until those fixes are released, use the exact tested source
commit already pinned in requirements.txt. Unpatched mjswan 0.12.12 cannot build this task.
The mjviser pin avoids installing `mujoco-core` 4 alongside mjswan's MuJoCo 3.11;
the script also detects a mismatched imported runtime.
The model package is pinned to the published 0.2.5 assets used in the verified pilot.

Open the printed loopback URL. `tasks/browser-walk/dist` is the static bundle;
`parity.json` records 64 Torch/ONNX action comparisons on native rollout states.
Every exported graph must pass the ONNX structural checker and native ORT session
creation. The bundle manifest hashes the actual reloaded model archive and all
shipped files, with ordered I/O, units, frozen normalization and runtime versions.
Integrity is checked before launching the local server. Static hosting needs its
own integrity/transport setup; the generic viewer does not consume this sidecar.

The task runs at **5 ms control**, with five 1 ms physics steps, as in training.
Joint-velocity observations are multiplied by control_dt; planar root velocities
remain m/s. The adapter supports the plain walking excitation pipeline and rejects
fatigue, motor noise, rerouting, custom stages and mixed actuator types. Heading
and reset are traced from the actual task. Browser physics already forwards derived
state, so the training-only sync-forward term is omitted.

Numerical parity is a software check. Watch complete browser rollouts and resets
before sharing the demo; it does not establish robust gait or human validity.
Muscle activation, input excitation, muscle force and EMG are different signals.
