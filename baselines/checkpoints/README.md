# Default-run policies (mjlab / RSL-RL)

Every mjlab-registered env id with a default `scripts/train_mjlab.py` run (Sarc/Fati/Reaf
muscle-condition variants excluded here; they use the same checkpoint as their base env).

**Deterministic success** = success rate of the mean-action policy, measured with
`scripts/eval_mjlab_policy.py --backend mjlab` (the GPU backend; the same checkpoints are also
loadable on the CPU backend, but the column was not measured there).

**`-` in both columns** = no checkpoint reached 25% deterministic success yet (either
unconverged, or not evaluated — see the notes below), so none is published as a default policy.

Checkpoints are hosted on
[myohub/myosuite-3-baselines](https://huggingface.co/myohub/myosuite-3-baselines) (Hugging Face)
rather than in this git repository, and are downloaded automatically by the tutorials and
`myosuite.utils.checkpoint_utils.find_checkpoint` the first time they're needed. Use one
directly::

    python scripts/eval_mjlab_policy.py <env_id> --backend cpu   # or mjlab

(pass `--checkpoint <run directory>` instead to use a checkpoint you trained yourself). Resume a
run with `python scripts/train_mjlab.py <env_id> --agent.resume True --agent.load-run <run> --env.scene.num-envs 4096`.

**Not all published policies converged to the standard 95% threshold**, and training for more
iterations may well reach higher success rates: `myoLegStandRandom-v0`/`myoHandPoseRandom-v0`
(36%, plateaued), `myoArmReachRandom-v0` (64%), `myoFingerReachRandom-v0` (48%),
`myoLegDirectionalRandom-v0` (30%) and `myoLegHillyTerrainWalk-v0` (25%).

**`myoChallengeChaseTagFBP2-v0` and `myoChallengeTableTennisP{0,1,2}-v0`** — `scripts/eval_mjlab_policy.py`
previously crashed on `--backend mjlab` for any task whose env config uses the standard mjlab
`"policy"` observation group name (it hardcoded `obs["actor"]`, a key that only exists for
MuscleMimic tasks, which register a redundant `"actor"` group as a workaround); fixed to read
`obs["policy"]`, the group every mjlab task config actually defines. `ChaseTagFBP2` now evaluates,
but the task has no `success`/metrics term configured yet, so deterministic success still can't be
measured until that's added. `TableTennisP{0,1,2}` construct and reset correctly under the fix, but
no trained checkpoint for them is available locally to run a full eval and publish a number.

| Env                          | Checkpoint         | Deterministic success (mjlab) |
| ---------------------------- | ------------------ | ----------------------------- |
| motorFingerPoseFixed-v0      | -                  | -                             |
| motorFingerPoseRandom-v0     | -                  | -                             |
| motorFingerReachFixed-v0     | -                  | -                             |
| motorFingerReachRandom-v0    | -                  | -                             |
| myoArmReachFixed-v0          | `model_696.pt`   | 100.0%                        |
| myoArmReachRandom-v0         | `model_4999.pt`  | 64.1%                         |
| myoChallengeChaseTagFBP2-v0  | -                  | -                             |
| myoChallengeTableTennisP0-v0 | -                  | -                             |
| myoChallengeTableTennisP1-v0 | -                  | -                             |
| myoChallengeTableTennisP2-v0 | -                  | -                             |
| myoElbowPose1D6MExoFixed-v0  | `model_9.pt`     | 100.0%                        |
| myoElbowPose1D6MExoRandom-v0 | `model_13.pt`    | 100.0%                        |
| myoElbowPose1D6MFixed-v0     | `model_100.pt`   | 100.0%                        |
| myoElbowPose1D6MRandom-v0    | `model_200.pt`   | 100.0%                        |
| myoFingerPoseFixed-v0        | `model_59.pt`    | 100.0%                        |
| myoFingerPoseRandom-v0       | `model_321.pt`   | 98.4%                         |
| myoFingerReachFixed-v0       | `model_45.pt`    | 100.0%                        |
| myoFingerReachRandom-v0      | `model_7900.pt`  | 48.4%                         |
| myoHandPose0Fixed-v0         | -                  | -                             |
| myoHandPose1Fixed-v0         | `model_357.pt`   | 100.0%                        |
| myoHandPose2Fixed-v0         | `model_416.pt`   | 100.0%                        |
| myoHandPose3Fixed-v0         | `model_801.pt`   | 100.0%                        |
| myoHandPose4Fixed-v0         | `model_410.pt`   | 100.0%                        |
| myoHandPose5Fixed-v0         | `model_486.pt`   | 100.0%                        |
| myoHandPose6Fixed-v0         | `model_473.pt`   | 100.0%                        |
| myoHandPose7Fixed-v0         | `model_542.pt`   | 100.0%                        |
| myoHandPose8Fixed-v0         | `model_370.pt`   | 100.0%                        |
| myoHandPose9Fixed-v0         | `model_459.pt`   | 100.0%                        |
| myoHandPoseFixed-v0          | -                  | -                             |
| myoHandPoseRandom-v0         | `model_24999.pt` | 35.9%                         |
| myoHandReachFixed-v0         | `model_97.pt`    | 100.0%                        |
| myoHandReachRandom-v0        | `model_1912.pt`  | 93.8%                         |
| myoLegDirectionalBackward-v0 | `model_687.pt`   | 100.0%                        |
| myoLegDirectionalForward-v0  | `model_692.pt`   | 96.9%                         |
| myoLegDirectionalRandom-v0   | `model_4999.pt`  | 29.7%                         |
| myoLegHillyTerrainWalk-v0    | `model_4999.pt`  | 25.0%                         |
| myoLegRoughTerrainWalk-v0    | `model_3715.pt`  | 90.6%                         |
| myoLegStairTerrainWalk-v0    | -                  | -                             |
| myoLegStandRandom-v0         | `model_4999.pt`  | 35.9%                         |
| myoLegWalk-v0                | `model_1355.pt`  | 98.4%                         |
| myoTorsoExoPoseFixed-v0      | `model_188.pt`   | 100.0%                        |
| myoTorsoPoseFixed-v0         | `model_103.pt`   | 100.0%                        |
