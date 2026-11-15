# Optional TERRA ADL apartment extension

This optional long-running extension of [tutorial 4.4](../../../../4.4_Programmatic_Waypoint_Tasks.ipynb) uses the same waypoint environment, reference planner, TERRA controller, and ghost renderer to walk a furnished split-level apartment.

It is a standalone tutorial example—not an installed package module, registered task, or CI/release test. Neither the apartment rollout nor its video rendering runs in GitHub Actions.

From the repository root, with `.[musclemimic]` and video dependencies installed:

```bash
python tutorials/files/4.4/adl_apartment/demo.py
```

The script downloads the public TERRA checkpoint if `TERRA_CHECKPOINT` is unset. Optional environment variables: `TERRA_ADL_OUT`, `TERRA_SEED`, and `TERRA_STOCHASTIC`. Headless Linux rendering uses `MUJOCO_GL=egl`.

Full controller/physics stepping is retained; only video capture is subsampled to 5 fps. Outputs include the actual policy rollout, a frame preview, and JSON waypoint metrics. This is not a validated ADL interaction benchmark (sitting/manipulation are not implemented).
