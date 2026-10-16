# MyoHalloween — Pumpkin Dodge Web Demo

Work-in-progress staging area for the Halloween browser demo and its policy-training/export pipeline.

## Goal

Run MyoFullBody in the browser and let the user throw pumpkins manually, or let the demo launch them automatically. A muscle-driven full-body controller should react to the predicted pumpkin trajectory and evade it.

## Architecture

- Browser physics/rendering: MuJoCo WASM via mjswan-compatible assets and ONNX Runtime Web.
- Character: MyoFullBody.
- Control output: 354 muscle controls.
- Threat input: pumpkin state plus a compact dodge command derived from trajectory prediction.
- Throw modes: manual and automatic.
- Policy artifact: `halloween_dodge.onnx` plus `halloween_dodge.json` metadata exported by the companion tutorial notebook.

## Development layout

- `tutorials/5.5_MuscleMimic_Halloween_Dodge.ipynb`: train/distill and export the policy.
- `docs/halloween_demo/web/`: browser demo source/staging files.
- `docs/halloween_demo/model/`: expected policy metadata / placeholder location for exported artifacts.

The browser task and training notebook share a fixed policy contract so they can be developed independently.
