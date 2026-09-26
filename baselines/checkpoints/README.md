# Default-run policies (mjlab / RSL-RL)

Newest checkpoint of the latest `scripts/train_mjlab.py` run of each env. Each folder
`<env_id>/model_<iter>.pt` works with both backends:

```bash
python scripts/eval_mjlab_policy.py <env_id> --checkpoint baselines/checkpoints/<env_id> --backend cpu   # or mjlab
```

and is found automatically by `tutorials/1.2_Load_Policy.ipynb` when no `logs/` run exists.
**Deterministic success** = success rate of the mean-action policy, measured with
`scripts/eval_mjlab_policy.py --backend mjlab` (the GPU backend; the same checkpoints
are also loadable on the CPU backend, but the column was not measured there). `-` = not
measured.

**Not all policies converged.** Several stopped below the 95% success threshold (see the
table), and training for more iterations may well reach higher success rates. Resume a
run with `python scripts/train_mjlab.py <env_id> --agent.resume True --agent.load-run <run>
--env.scene.num-envs 4096`.

**Caveat — `myoLeg{Walk,DirectionalForward,DirectionalBackward,DirectionalRandom}-v0`:**
trained (2026-09-25) before the leg twins got their success metric / shared PPO defaults.
On the CPU backend they walk (Directional Forward/Backward: 100% success), but on the
current mjlab twin they fall after ~200 steps, so they are provisional; retrain them
with `python scripts/train_mjlab.py <env_id> --env.scene.num-envs 4096` and refresh these files.
`myoHandPose0Fixed`/`myoHandPoseFixed` (0%), `myoLegStandRandom` (36%, plateaued) and the
reach/motor envs below 95% are unconverged snapshots. `myoTorsoExoPoseFixed` (converged) also
runs on the CPU env (same return and success as on mjlab).

| Env | Checkpoint | Deterministic success (mjlab) |
|---|---|---|
| motorFingerPoseFixed-v0 | model_9998.pt | 0.0% |
| motorFingerPoseRandom-v0 | model_9998.pt | 15.6% |
| motorFingerReachFixed-v0 | model_9998.pt | 0.0% |
| motorFingerReachRandom-v0 | model_9998.pt | 9.4% |
| myoArmReachFixed-v0 | model_696.pt | 100.0% |
| myoArmReachRandom-v0 | model_4999.pt | 64.1% |
| myoElbowPose1D6MExoFixed-v0 | model_9.pt | 100.0% |
| myoElbowPose1D6MExoRandom-v0 | model_13.pt | 100.0% |
| myoElbowPose1D6MFixed-v0 | model_100.pt | 100.0% |
| myoElbowPose1D6MRandom-v0 | model_200.pt | 100.0% |
| myoFingerPoseFixed-v0 | model_59.pt | 100.0% |
| myoFingerPoseRandom-v0 | model_321.pt | 98.4% |
| myoFingerReachFixed-v0 | model_45.pt | 100.0% |
| myoFingerReachRandom-v0 | model_7900.pt | 48.4% |
| myoHandPose0Fixed-v0 | model_3500.pt | 0.0% |
| myoHandPose1Fixed-v0 | model_357.pt | 100.0% |
| myoHandPose2Fixed-v0 | model_416.pt | 100.0% |
| myoHandPose3Fixed-v0 | model_801.pt | 100.0% |
| myoHandPose4Fixed-v0 | model_410.pt | 100.0% |
| myoHandPose5Fixed-v0 | model_486.pt | 100.0% |
| myoHandPose6Fixed-v0 | model_473.pt | 100.0% |
| myoHandPose7Fixed-v0 | model_542.pt | 100.0% |
| myoHandPose8Fixed-v0 | model_370.pt | 100.0% |
| myoHandPose9Fixed-v0 | model_459.pt | 100.0% |
| myoHandPoseFixed-v0 | model_4999.pt | 0.0% |
| myoHandReachFixed-v0 | model_97.pt | 100.0% |
| myoHandReachRandom-v0 | model_1912.pt | 93.8% |
| myoLegDirectionalBackward-v0 | model_4999.pt | - |
| myoLegDirectionalForward-v0 | model_4999.pt | - |
| myoLegDirectionalRandom-v0 | model_4999.pt | - |
| myoLegStandRandom-v0 | model_4999.pt | 35.9% |
| myoLegWalk-v0 | model_4999.pt | - |
| myoTorsoExoPoseFixed-v0 | model_227.pt | 100.0% |
| myoTorsoPoseFixed-v0 | model_103.pt | 100.0% |
