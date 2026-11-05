# Optional TERRA apartment demo

This is an optional, long-running demonstration of the waypoint/TERRA integration in a furnished split-level apartment. It is **not** a supported registered task, a package module, or a CI/release gate.

The example reuses MyoSuite's `build_terrain_fullbody_spec`, `plan_waypoint_reference`, `TerrainController`, `myoFullBodyWaypoint-v0`, and ghost visualization. It runs independently and downloads the public TERRA checkpoint when not available locally.

Run locally or in Colab from the repository root with the optional MuscleMimic dependencies installed:

```bash
python examples/terra_adl/demo.py
```

`MUJOCO_GL=egl` supports headless rendering. Set `TERRA_CHECKPOINT` to a cached checkpoint, `TERRA_ADL_OUT` for output location, or `TERRA_SEED`/`TERRA_STOCHASTIC` to reproduce a rollout. This is intentionally excluded from GitHub Actions because full TERRA inference and rendering are expensive. A video is a policy rollout, distinct from the generated motion reference.
