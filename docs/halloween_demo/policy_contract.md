# MyoHalloween policy contract

The web demo and training notebook exchange two files:

- `halloween_dodge.onnx`
- `halloween_dodge.json`

## Inputs

The initial contract is intentionally compact and browser-friendly. The policy consumes a flat float32 vector:

1. `qpos_reduced`: selected root and joint positions.
2. `qvel_reduced`: selected root and joint velocities.
3. `muscle_state`: selected/aggregated muscle activation and force features.
4. `command`: 8 values
   - escape_x
   - escape_y
   - duck
   - torso_lean_x
   - torso_lean_y
   - urgency
   - time_to_contact
   - threat_height
5. `prev_action`: previous 354-dimensional muscle action.

The exact feature ordering and normalization statistics are stored in the JSON metadata. The browser must not hard-code training-time normalization.

## Output

`action[354]`, float32, clipped to `[-1, 1]` and written directly to MyoFullBody muscle controls.

## Throw-independent interface

The policy never receives mouse state or an auto/manual flag. Both throw modes produce the same world-state threat estimate. The browser trajectory predictor converts the pumpkin state into the 8-dimensional command.

## Metadata

`halloween_dodge.json` includes:

```json
{
  "format_version": 1,
  "model": "MyoFullBody",
  "action_dim": 354,
  "input_dim": 0,
  "feature_order": [],
  "obs_mean": [],
  "obs_std": [],
  "control_hz": 50,
  "onnx_file": "halloween_dodge.onnx"
}
```

`input_dim`, `feature_order`, and normalization arrays are populated by the export notebook.
