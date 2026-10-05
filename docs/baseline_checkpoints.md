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
iterations may well reach higher success rates: `myoHandPoseRandom-v0` (36%, plateaued),
`myoArmReachRandom-v0` (80%), `myoFingerReachRandom-v0` (73%), `motorFingerReachRandom-v0` (94%),
`myoLegDirectionalRandom-v0` (54%), `myoLegStandRandom-v0` (34%) and `myoLegHillyTerrainWalk-v0`
(27%). `myoLegDirectionalRandom-v0` succeeds with exploration noise (99% sampled success) but only about half of the time with the mean
action; the checkpoint is the best one of the run.

`myoFingerReachRandom-v0` and `motorFingerReachRandom-v0` used to sample their targets uniformly in a box
of which only about 55% lies within the fingertip's workspace, which capped any policy near that. They now
sample only targets the fingertip can reach, and their policies were retrained.

`myoArmReachRandom-v0` started 16.5% of its episodes beyond its far threshold (`far_th` = 1.0 m): the targets in
the upper part of its box lay more than 1 m from the hanging fingertip, and those episodes ended at step 2 unless
the policy closed the gap in its first two steps. Its far threshold is now 1.3 m. The checkpoint below was
trained with the old threshold and re-evaluated with the new one. On the CPU (the same 200 seeds) its
deterministic success rises from 67.0% to 76.5%: it solves 85% of the targets it used to be cut off from (27%
before) and is unchanged on the others (74.9%). On mjlab (64 episodes) it reaches 79.7%, against
71.9% with the old threshold in the same setup.

`myoHandReachRandom-v0` started 97% of its episodes beyond its far threshold (`far_th` = 0.034, 0.17 m over the
five fingertips) from the open hand, and those episodes ended at step 2 unless the policy closed the hand within
the first 40 ms. Its far threshold is now 0.075. The checkpoint below was trained with the old threshold and learned to close the hand
that fast, so it is not affected: re-evaluated with the new threshold it gives the same 96.9% on mjlab (the
64 episodes are identical with either threshold) and 92.0% over 500 CPU episodes (91.8% with the old one).

**Compatibility with the ms3 observation changes.** Policies are tied to the observation contract they
were trained with. These changes since the first set of checkpoints affect them:

- The directional-leg twins observed raw `qvel`; every backend now observes `qvel * ctrl_dt`.
  The old `myoLegDirectional{Forward,Backward,Random}-v0` checkpoints no longer work (0% success)
  and were retrained.
- The `motorFinger*` envs now use motors with four times the stock gear (80/20/20/40/40): with the stock
  gears (and with x2) no policy left 0% success, with x4 the three fixed/pose tasks reach 100%.
  These policies are new.
- Observations are no longer clipped to +-10 and every derived quantity (muscle force, sensors, `cvel`)
  is refreshed after each step (walk, stand and reach twins, 55 CPU env ids). The walk, terrain and
  reach checkpoints above were re-evaluated after this change; their success rates are the numbers in
  the table. `myoLegStandRandom-v0` dropped from 35.9% to 23.4% with the old checkpoint and was retrained (34.1%).
- Later changes touch only tasks without a published baseline here: the CPU `myoChallengeChaseTagFBP2-v0`
  now uses the 537-dim `chasetag_obs` layout, 0.01 s steps and flat ground (the mjlab task takes the CPU
  rewards, reset and opponent); the mjlab TableTennis scene and scoring and the physics options of the
  TableTennis and MuscleMimic mjlab tasks follow the CPU models; the MuscleMimic bridge maps outputs with
  `clip(a, 0, 1)`. Policies trained on the earlier versions of those tasks must be retrained or
  re-evaluated; the checkpoints in the table are not affected.

Re-evaluate a checkpoint after any change to an env's observations with
`python scripts/eval_mjlab_policy.py <env_id> --backend mjlab`; train it again if its success rate drops
noticeably (with 64 episodes, differences of a few points are noise). Success rates were measured with
64 parallel episodes of the deterministic policy.

**`myoChallengeChaseTagFBP2-v0`** (success: the agent tags the opponent) and
**`myoChallengeTableTennisP{0,1,2}-v0`** have no trained checkpoint available yet to evaluate.

| Env                          | Checkpoint         | Deterministic success (mjlab) |
| ---------------------------- | ------------------ | ----------------------------- |
| motorFingerPoseFixed-v0      | `model_267.pt`     | 100.0%                        |
| motorFingerPoseRandom-v0     | `model_152.pt`     | 100.0%                        |
| motorFingerReachFixed-v0     | `model_120.pt`     | 100.0%                        |
| motorFingerReachRandom-v0    | `model_1999.pt`    | 93.9%                         |
| myoArmReachFixed-v0          | `model_696.pt`   | 100.0%                        |
| myoArmReachRandom-v0         | `model_4999.pt`    | 79.7%                         |
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
| myoFingerReachRandom-v0      | `model_1500.pt`    | 72.7%                         |
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
| myoHandReachRandom-v0        | `model_1912.pt`    | 96.9%                         |
| myoLegDirectionalBackward-v0 | `model_583.pt`     | 96.9%                         |
| myoLegDirectionalForward-v0  | `model_681.pt`     | 100.0%                        |
| myoLegDirectionalRandom-v0   | `model_2000.pt`    | 53.8%                         |
| myoLegHillyTerrainWalk-v0    | `model_4999.pt`    | 26.6%                         |
| myoLegRoughTerrainWalk-v0    | `model_3715.pt`    | 89.1%                         |
| myoLegStairTerrainWalk-v0    | -                  | -                             |
| myoLegStandRandom-v0         | `model_3500.pt`    | 34.1%                         |
| myoLegWalk-v0                | `model_1355.pt`    | 100.0%                        |
| myoTorsoExoPoseFixed-v0      | `model_188.pt`   | 100.0%                        |
| myoTorsoPoseFixed-v0         | `model_103.pt`   | 100.0%                        |
