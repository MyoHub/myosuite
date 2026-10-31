# TERRA parity fixtures

`terra_tracking_goals.npz` checks the pinned TERRA full-body tracking observation:
six nonzero root, joint and velocity perturbations, including bounded future
horizons at the end of a reference. The fixture stores upstream commit hashes.
The generator executes selected upstream definitions, including complete tracking
goal assembly and MuscleMimic site velocities, without importing the packages.
It needs only MyoSuite's optional actor assets, NumPy, SciPy and MuJoCo.

With local upstream checkouts at the commits specified in the generator:

```bash
python scripts/terra_parity/tracking_fixture.py /path/to/terra /path/to/musclemimic myosuite/tests/data/terra_tracking_goals.npz
```

`terra_upstream_obs.npz` is an older 15-step native upstream rollout. Its replay
checks controller physics and the physical observation prefix; the obsolete compact
goal is excluded. Regenerating that historical fixture requires the upstream
Python >=3.11 environment, the MuscleMimic commit pinned by TERRA,
`musclemimic_models==1.0.5`, and the TERRA checkout on `PYTHONPATH`:

```bash
python scripts/terra_parity/make_reference.py ref.npz
python scripts/terra_parity/upstream_rollout.py ref.npz myosuite/tests/data/terra_upstream_obs.npz
```
