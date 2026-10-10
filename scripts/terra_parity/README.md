# TERRA observation parity fixture

`myosuite/tests/data/terra_upstream_obs.npz` holds a short TERRA rollout computed by the upstream
TERRA / MuscleMimic code; `myosuite/tests/test_terra.py` replays it through
`myosuite.integrations.terra` and requires the same states and observations.

Regenerate it after a change to the TERRA actor or observation:

1. MyoSuite environment: `python scripts/terra_parity/make_reference.py ref.npz`
2. Upstream environment (Python >= 3.11, the JAX/MuJoCo pins of
   [musclemimic](https://github.com/amathislab/musclemimic) at TERRA's commit, `musclemimic_models==1.0.5`, with
   the musclemimic and [TERRA](https://github.com/amathislab/terra) `src` checkouts on `PYTHONPATH`):
   `python scripts/terra_parity/upstream_rollout.py ref.npz myosuite/tests/data/terra_upstream_obs.npz`
