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

Re-evaluate a checkpoint after any change to an env's observations with
`python scripts/eval_mjlab_policy.py <env_id> --backend mjlab`; train it again if its success rate drops
noticeably (with 64 episodes, differences of a few points are noise). Success rates were measured with
64 parallel episodes of the deterministic policy.

**Terrain and CPU evaluation.** A deterministic CPU evaluation is a single trajectory. The mjlab twins of the
rough, hilly and stairs walk tasks bake one terrain into the height field (Warp shares one `hfield_data` between
worlds), while the CPU envs draw a new terrain at every reset, so CPU and mjlab success rates on these ids
measure different things. Tiling several height-field patches in one field, one per env, would match the CPU
distribution; this is not implemented.

**Older policies and datasets.** Policies trained on the earlier CPU `myoChallengeChaseTagFBP2-v0` contract
(1496-dim observation, 0.02 s control step, random terrain) no longer load, and SAR activation datasets or synergy
models collected before the MuscleMimic bridge fix (#459) must be recollected.

**Collision bounds and inertia of randomized objects.** Die-reorient, Baoding P2 and weighted elbow-pose episodes now use refreshed collision bounds and mass-consistent inertia, so their dynamics can differ from earlier builds; re-evaluate policies and recollect affected datasets.

**MuscleMimic full body (single clip).** `checkpoints/myoMimicFullbody-v0-walking_medium06/model_81380.pt` on the
Hugging Face repo imitates the clip `KIT/167/walking_medium06` (2.0B steps, 1024 envs, about 30 h on one GPU). It
is scored by clip completion instead of success: 86% of 576 mean-action episodes from random start frames play to
the end of the clip (84-88% per run), the rest end on a pose deviation. It is an mjlab-only baseline (no CPU twin: the
CPU `myoMimicFullbody-v0` uses random targets), so it sits in its own folder and is loaded with `--checkpoint` and
`MIMIC_CLIP` (tutorial 5.3).

**`myoChallengeChaseTagFBP2-v0`** (success: the agent tags the opponent) and
**`myoChallengeTableTennisP{0,1,2}-v0`** have no trained checkpoint available yet to evaluate.

| Env                          | Checkpoint         | Deterministic success (mjlab) |
| ---------------------------- | ------------------ | ----------------------------- |
| motorFingerPoseFixed-v0      | `model_267.pt`   | 100.0%                        |
| motorFingerPoseRandom-v0     | `model_152.pt`   | 100.0%                        |
| motorFingerReachFixed-v0     | `model_120.pt`   | 100.0%                        |
| motorFingerReachRandom-v0    | `model_1999.pt`  | 93.9%                         |
| myoArmReachFixed-v0          | `model_696.pt`   | 100.0%                        |
| myoArmReachRandom-v0         | `model_4999.pt`  | 79.7%                         |
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
| myoFingerReachRandom-v0      | `model_1500.pt`  | 72.7%                         |
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
| myoHandReachRandom-v0        | `model_1912.pt`  | 96.9%                         |
| myoLegDirectionalBackward-v0 | `model_583.pt`   | 96.9%                         |
| myoLegDirectionalForward-v0  | `model_681.pt`   | 100.0%                        |
| myoLegDirectionalRandom-v0   | `model_2000.pt`  | 53.8%                         |
| myoLegHillyTerrainWalk-v0    | `model_4999.pt`  | 26.6%                         |
| myoLegRoughTerrainWalk-v0    | `model_3715.pt`  | 89.1%                         |
| myoLegStairTerrainWalk-v0    | -                  | -                             |
| myoLegStandRandom-v0         | `model_3500.pt`  | 34.1%                         |
| myoLegWalk-v0                | `model_1355.pt`  | 100.0%                        |
| myoTorsoExoPoseFixed-v0      | `model_188.pt`   | 100.0%                        |
| myoTorsoPoseFixed-v0         | `model_103.pt`   | 100.0%                        |
