# Default-run policies (mjlab / RSL-RL)

**Waypoint reset reward fix (3.1.0):** mjlab `myoFullBodyWaypoint-v0` now preserves the initial route distance after repeated resets. The first progress reward of each new episode uses the actual distance change instead of a spurious negative distance. Observation/action shapes and checkpoint loading are unchanged; returns from runs with the old reset bug should be re-evaluated. CPU behavior is unchanged.

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

| Env                          | Checkpoint         | Deterministic success (mjlab) |
| ---------------------------- | ------------------ | ----------------------------- |
| motorFingerPoseFixed-v0      | `model_83.pt`      | 100.0%                        |
| motorFingerPoseRandom-v0     | `model_1499.pt`    | 99.6%                         |
| motorFingerReachFixed-v0     | `model_57.pt`      | 100.0%                        |
| motorFingerReachRandom-v0    | `model_1499.pt`    | 100.0%                        |
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
| myoLegDirectionalBackward-v0 | `model_681.pt`   | 100.0%²                       |
| myoLegDirectionalForward-v0  | `model_583.pt`   | 96.9%²                        |
| myoLegDirectionalRandom-v0   | `model_2000.pt`  | 53.8%                         |
| myoLegHillyTerrainWalk-v0    | `model_4999.pt`  | 0%³ (20.7% with random hill heights) |
| myoLegRoughTerrainWalk-v0    | `model_4176.pt`  | 91.0%¹                        |
| myoLegStairTerrainWalk-v0    | -                  | -                             |
| myoLegStandRandom-v0         | `model_3500.pt`  | 34.1%                         |
| myoLegWalk-v0                | `model_3000.pt`  | 100.0%⁴ (**incompatible with 3.1.0**: 0% from its reset) |
| myoTorsoExoPoseFixed-v0      | `model_188.pt`   | 100.0%                        |
| myoTorsoPoseFixed-v0         | `model_103.pt`   | 100.0%                        |

¹ Resumed on MyoSuite 3.0.0 from a run trained with mjlab 1.3.

² `myoLegDirectionalForward-v0` and `myoLegDirectionalBackward-v0` swapped names after 3.0.0: the model faces `-y`, so the task that
walks along `-y` is now the forward one. Each checkpoint moved with its task definition; a checkpoint downloaded under the old
name walks in the opposite direction of the new one.

³ The registered env is deterministic: `variant="fixed"` (hill height 0.63) and a fixed start, so the mean-action policy plays a single trajectory, which falls at step 673 (CPU, 3.1 start). With `task_kwargs={"variant": "random"}` (hill height drawn from U(0.53, 0.73) at every reset), 256 CPU episodes give 20.7% (95% Wilson interval 16.2-26.1%) from the 3.1 start and 9.4% (128 episodes) from the old start; sampled actions on the registered env give 43.0% (128 episodes). The earlier 26.6% (one draw of 64 episodes) and 15.3% (mean of seeds 1-3) were measured on the mjlab twin, whose variation between seeds is not explained yet; the released 3.0.0 code gives 18.4, 19.9 and 14.1% on the same seeds, so 3.1 did not regress.

⁴ **Incompatible with MyoSuite 3.1.0.** Measured before 3.1 from the old start. From 3.1 the leg tasks reset with the coupled knee joints on their polynomials (#630), and this checkpoint falls from that start (0% success, *Coupled knee joints at reset (leg tasks)* below). A checkpoint trained from the new start follows in 3.2.

CPU evaluations that pass a NumPy array as `FatigueWrapper.fatigue_reset_vec` now reuse the requested initial fatigue in every episode. Earlier versions mutated that array during stepping, which changed later episodes. Reevaluate checkpoints measured with this configuration.

`myoChallengeChaseTagP2-v0`, `myoChallengeChaseTagP2eval-v0` and their `myoSarc` / `myoFati` variants pay the EVADE win again: the last step of an EVADE episode that reaches the time limit has `solved = 1` (+1000) and score 1.0. In 3.1.0 that win could not fire, so EVADE returns and `get_metrics` scores measured on 3.1.0 are too low for policies that survive; reevaluate them. Observations, actions and dynamics are unchanged, and no checkpoint of these ids is published here.

## Compatibility notes for 3.1.0

What changes for checkpoints between MyoSuite 3.0.x and 3.1.0. The notes below the table give the details.

- **Hugging Face revisions.** MyoSuite 3.1 reads the baselines repo at tag `v3.1.0`, then `v3.1`, and falls back to
  `main` when no tag exists or the file is not in it. 3.0.x installs keep reading `v3.0`.
- **`motorFinger*`.** The four checkpoints in the table are a new round for the `myo_sim` 0.2.4 gears, trained with
  the bounded action std of #584: `motorFingerPoseFixed-v0` `model_83.pt`, `motorFingerPoseRandom-v0`
  `model_1499.pt`, `motorFingerReachFixed-v0` `model_57.pt`, `motorFingerReachRandom-v0` `model_1499.pt`. They are on
  `main` and go into `v3.1.0`; the `v3.0` checkpoints match only the earlier gears (*Motor finger gears* below).
- **`myoChallengeChaseTagFBVs-v0` and `myoChallengeChaseTagFBP2-v0`** run on `myo_sim`'s `myofullbody` with 10 s
  episodes. Policies trained on the `musclemimic_models` body before 3.1 do not transfer reliably; the 1v1 baseline is
  `onnx/chasetag_fbvs_baseline.onnx` (*`myoChallengeChaseTagFBVs-v0`* below).
- **Leg resets on the knee couplings (#630).** `myoLegWalk-v0`, the terrain walks, `myoLegStandRandom-v0` and the
  directional walks start with the coupled knee joints on their polynomials. The published `myoLegWalk-v0` checkpoint
  was trained from the old start and falls from the new one; the rough-terrain checkpoint is unaffected (*Coupled knee
  joints at reset (leg tasks)* below). The MuscleMimic full-body envs reset the same way.
- **`myoMimic-v0`** is new. Its clips are matched to the body by a kinematic fingerprint, so clips keep
  loading under `myo_sim` releases that leave the body's kinematics unchanged; on a mismatch the error names the
  `myo_sim` version the clips were fitted for. No `myoMimic-v0` checkpoint is published in this repo.
- **Joint couplings and fatigue in `myo_sim` 0.2.5.** They moved from MyoSuite into `myo_sim` with identical numerics,
  so no checkpoint changes.
- **Training defaults.** The bounded action std (#566) and the fatigue groups and exact fatigue step of the
  `myoFati*` variants change training and fatigue results, not the published base-env checkpoints (notes below).

---

**Not all published policies converged to the standard 95% threshold**, and training for more
iterations may well reach higher success rates: `myoHandPoseRandom-v0` (36%, plateaued),
`myoArmReachRandom-v0` (80%), `myoFingerReachRandom-v0` (73%),
`myoLegDirectionalRandom-v0` (54%), `myoLegStandRandom-v0` (34%) and `myoLegHillyTerrainWalk-v0`
(15%). `myoLegDirectionalRandom-v0` succeeds with exploration noise (99% sampled success) but only about half of the time with the mean
action; the checkpoint is the best one of the run.

Re-evaluate a checkpoint after any change to an env's observations with
`python scripts/eval_mjlab_policy.py <env_id> --backend mjlab`; train it again if its success rate drops
noticeably (with 64 episodes, differences of a few points are noise). Success rates were measured with
64 parallel episodes of the deterministic policy.

**Terrain and CPU evaluation.** A deterministic CPU evaluation is a single trajectory. The mjlab twins of the
rough, hilly and stairs walk tasks bake one terrain into the height field (Warp shares one `hfield_data` between
worlds), while the CPU rough-terrain walk draws a new terrain at every reset, so CPU and mjlab success rates on that id
measure different things. The hilly and stairs walks are registered with one fixed terrain (`variant="fixed"`) on both backends. Tiling several height-field patches in one field, one per env, would match the CPU
distribution; this is not implemented.

**Older policies and datasets.** Policies trained on the earlier CPU `myoChallengeChaseTagFBP2-v0` contract
(1496-dim observation, 0.02 s control step, random terrain) no longer load, and SAR activation datasets or synergy
models collected before the MuscleMimic bridge fix (#459) must be recollected.

**Collision bounds and inertia of randomized objects.** Die-reorient, Baoding P2 and weighted elbow-pose episodes now use refreshed collision bounds and mass-consistent inertia, so their dynamics can differ from earlier builds; re-evaluate policies and recollect affected datasets.

**Motor finger gears.** With `myo_sim` 0.2.4 the `motorFinger*` envs use the gears of `motorfinger_v0.xml` as they are, 160/80/80/160/160 (`extn`, `adabR`, `adabL`, `mflx`, `dflx`), and MyoSuite no longer scales them. Checkpoints trained on the earlier 80/20/20/40/40 no longer apply. Their observations, actions and control step are unchanged, so they still load and the env contract hash in `manifest.json` does not warn about them: use the checkpoints of the release that matches your MyoSuite version. The four `motorFinger*` rows are means over seeds 0-2 with 256 episodes each.

**Fatigue parameters of torso, hamstring, forearm and two-agent muscles.** In the `myoFati*` variants, lumbar and
abdominal muscles use the new `Trunk` row and hamstrings the `Thigh-Hamstring` row (both used `Default`), the forearm
rotators `SUP`/`PT`/`PQ` the `Elbow` row and `APL` the wrist-extensor row, and the left arm of the MuscleMimic models
and every muscle of two-agent scenes get their muscle group's row. The checkpoints above (base envs) are unchanged; fatigue results of these variants from earlier
builds are not comparable.

**Exact fatigue step.** The 3CC-r fatigue state of the `myoFati*` variants is now integrated exactly over each control step
(`MA` over the step instead of its start value). `MA` and `MF` change by up to about 4e-4 per step and the fatigue a policy
accumulates differs by a few percent of `MF` under fast-changing commands. The checkpoints above (base envs) are unchanged;
fatigue results of these variants from earlier builds are not bit-identical.

**MuscleMimic full body (single clip).** `checkpoints/myoMimicFullbody-v0-walking_medium06/model_81380.pt` on the
Hugging Face repo imitates the clip `KIT/167/walking_medium06` (AMASS / KIT motion capture retargeted with GMR; cite AMASS, KIT and MuscleMimic, see the *References* page; 2.0B steps, 1024 envs, about 30 h on one GPU). It
is scored by clip completion instead of success: 86% of 576 mean-action episodes from random start frames play to
the end of the clip (84-88% per run), the rest end on a pose deviation. It is an mjlab-only baseline (no CPU twin: the
CPU `myoMimicFullbody-v0` uses random targets), so it sits in its own folder and is loaded with `--checkpoint` and
`MIMIC_CLIP` (tutorial 5.3).

**Coupled knee joints at reset (MuscleMimic models).** The CPU envs `myoMimicFullbody-v0`, `myoFullBodyWaypoint-v0`
and `myoFullBodyDirectional-v0` and their mjlab tasks now start with the tibia and patella joints of each knee on their
joint couplings. From `qpos0` and the keyframe the patella started up to 5.2 cm off them and was pulled back over the
first 0.3 s. Clip starts (RSI, the directional env) were further off: the retargeted clips keep the
`musclemimic_models` 1.0.5 sign of the left knee's `rotation3` and `translation2` couplings, up to 0.5 rad off on 1.0.6.
The observation, action, control step and reward definitions are unchanged, so the env contract hash in `manifest.json`
does not warn and the checkpoints still load. Only the first steps of an episode differ, but re-evaluate a policy before
comparing it with new runs. The single-clip policy above was trained with the unprojected clip starts and was not
re-evaluated after this change.

**Lookahead at the clip end (MuscleMimic clip mode, after 3.1.0).** The lookahead observation of the clip-mode mjlab
tasks (`myoMimicFullbody-v0`, `myoMimicBimanual-v0`, the `myoMuscleMimic*` aliases, the SAR tasks) and of the CPU
`MuscleMimicClipEnvV0` (tutorial 5.2) now holds the last frame for slots past the clip end; it used to wrap to the
clip start. The observation size and the reward are unchanged, so the checkpoints still load, but within 100 frames
of a clip end (20 frames apart, 5 slots) a policy now sees other lookahead values. The single-clip policy above and
policies trained with tutorials 5.2 and 5.3 saw the wrapped values during training; they were not re-evaluated.

**Coupled knee joints at reset (leg tasks).** `myoLegWalk-v0`, the rough, hilly and stairs walks,
`myoLegStandRandom-v0` and the directional walks now start, on both backends, with the tibia and patella joints of
each knee on their joint couplings. Walk keyframe 2 set the bent knee's patella rotation 0.74 rad off its coupling
(MuJoCo pulled it back within about 0.2 s); the stand reset noise moved the coupled joints up to 0.18 rad, and the
directional keyframe is 3.5e-5 off. The observation, action, control step and reward definitions are unchanged, so
the env contract hash in `manifest.json` does not warn and the checkpoints still load, but they were trained with the
old start. Re-evaluated on the mjlab twin (CPU device, 64 mean-action episodes, `--seed 0`), old start -> new start:

| Env | Success | Mean episode length |
| --- | --- | --- |
| `myoLegWalk-v0` | 100% -> 0% | 1000 -> 233 |
| `myoLegRoughTerrainWalk-v0` | 100% -> 100% | 1000 -> 1000 |
| `myoLegHillyTerrainWalk-v0` | 0% -> 93.8% | 295 -> 957 |
| `myoLegStandRandom-v0` | 35.9% -> 32.8% | 136.9 -> 136.9 |

The walk policies were trained from one deterministic start. The walk policy depends on that start, not only on its
exact state: with N(0, σ) noise on every joint (CPU env, 24 starts each) it solves 16, 13 and 4 of 24 runs from the old
keyframe at σ = 0.005, 0.01 and 0.02 rad, and 0, 4 and 3 of 24 from the new start (without noise it falls at step 232
there). The rough-terrain policy, from the same keyframe, is unaffected, and the hilly policy goes the other way on this
seed (on the CPU with random hill heights it goes from 9.4% to 20.7%, footnote ³). Retrain or fine-tune the `myoLegWalk-v0` checkpoint before using it
with this version. The directional walks, whose start moved by 3.5e-5, were not re-evaluated.

**Bounded action std in the PPO defaults (#566).** The mjlab PPO configs of the muscle tasks and of `motorFinger*`
(whose actions are clipped before the linear map to `ctrlrange`) now keep the policy's action std in [0.05, 1.0]. The
std parameter is projected onto that range before every forward pass and stays trainable at the bounds. MuscleMimic
(with the SAR tasks of tutorial 5.5, which register with its config), ChaseTag FBP2, table tennis and the elbow config
also start at std 0.5 and use an entropy bonus of 0.001 (were 1.0 and 0.01). The bound limits the drift of the std of
saturated controls (up to the learning rate per update, less with the noise of PPO's surrogate gradient); it does not
by itself prevent saturated (bang-bang) controls. The checkpoints above, except the four `motorFinger*` ones, were
trained with the earlier, unbounded defaults. They still load, and `eval_mjlab_policy.py` scores them as before, with or without `--stochastic`: the bound
adds no parameter, and a run whose `params/agent.yaml` has no `std_range` is sampled with the checkpoint's own std
(runs trained with the bound are sampled with the std clamped to their range, as in training). Resuming training from
one of them projects a std above 1 to 1 on the first update (the MuscleMimic run of #566 ended at 14), and the std stays
trainable. A run with the new defaults is a new training setup, so its policy and success rate can differ from the
published ones. This includes pose, reach, leg, torso and `motorFinger*`, whose std was unbounded before (an elbow-pose
run without the bound took its smallest std below 0.05 after 220 iterations). The run times and results quoted in
tutorial 5.5 were measured with the earlier defaults.

**`myoChallengeChaseTagFBP2-v0`** (success: the agent tags the opponent) and
**`myoChallengeTableTennisP{0,1,2}-v0`** have no trained checkpoint available yet to evaluate.

**`myoChallengeChaseTagFBVs-v0`** (the scored 1v1 game, CPU only). `onnx/chasetag_fbvs_baseline.onnx` in the `myohub/myosuite-3-baselines` Hugging Face repo (`myosuite.core.hf_io.download_baseline_file`) plays one player, 546 -> 354, in either role. It is a starting point: it reaches and tags a player that does nothing, but it often falls before the 10 s limit: at about 6.7 s as a chaser against itself. Measured with action noise 0.02 and 12 episodes per role, on myo_sim 0.2.4 and MuJoCo 3.11; myo_sim 0.2.3 and MuJoCo 3.17 gave the same tags and falls:

| Match | Tags | Chaser falls | Mean length | Fell at (median) |
|---|---:|---:|---:|---|
| Chases a player that does nothing, as player A | 12/12 | 0 | 3.0 s | - |
| Chases a player that does nothing, as player B | 12/12 | 0 | 3.6 s | - |
| Against itself, A chases | 11/12 | 0 | 5.2 s | runner 1.7 s |
| Against itself, B chases | 0/12 | 12 | 7.1 s | runner 5.5 s, chaser 6.7 s |

It was fine-tuned with self-play on `myofullbody` (4,000 iterations, 2048 envs) from a policy that descends from the behaviour-cloned walker from the earlier version of tutorial 5.4 (`policy_bc_best.pt`, a student of MuscleMimic `amathislab/mm-10m-2`), so it is not trained from scratch on this body. Observations and actions match the game; only the dynamics of `myofullbody` were new to it. Policies trained on `musclemimic_models` do not carry over reliably: moving between releases 1.0.6 and 1.0.5 of that body already took a similar policy to 0 tags.

**Tutorial files for ChaseTag and directional walking (myoSuperMimic, `myofullbody`).** Tutorials 5.4, 5.6 and 5.7 download these from the same repo; they come from `training/chasetag_teacher` and were trained only on the KIT clips of `myoMimic-v0`'s default list (MuscleMimic's 972 KIT training motions, retargeted onto `myofullbody`; AMASS licence). Matches as above, 48 episodes per match and side:

| File | What | Tags a player that does nothing | Wins head-to-head vs the FBVs baseline |
|---|---|---:|---:|
| `supermimic/tracker_myofullbody_kit_30min.pt` | the myoSuperMimic tracker of the steered teacher (30-minute run; deployed policy only) | 92 % (teacher) | 71 % (teacher) |
| `directional/walker_myofullbody.pt` | the tutorial 5.4 walker (DAgger, 528 -> 354), as a player through `export_chasetag_onnx` | 100 % | 64 % |
| `onnx/chasetag_supermimic_student.onnx` | the DAgger student of the teacher (546 -> 354, yaw-canonicalised) | 98 % | 61 % |
| `onnx/chasetag_supermimic_selfplay.onnx` | a self-play refinement of that student (tutorial 5.6's stronger start) | 100 % | 76 % |

With tutorial 5.4's protocol (8 headings x 10 seeds from standing) the walker survives 91 % of the 2 s rollouts (alignment +0.85) and 85 % over 10 s (alignment +0.93).

**Manifests and releases.** Every `checkpoints/<env_id>/` folder on the Hugging Face repo has a `manifest.json` with the env id, a hash of the env contract (observation and action shapes, control step), the MyoSuite version and commit, and the success rate above. When a downloaded checkpoint is used, MyoSuite compares that hash with the current env and warns if they differ, because the policy may then not work. The repo is tagged per release (`v3.0`, `v3.1.0`, ...), and per patch release where baselines changed (`v3.0.1`); `download_baseline_checkpoint` (one env) and `download_baseline_file` (a single file, such as the MuscleMimic policy) take the tag of your installed version, the patch tag first (`v3.1.0`, then `v3.1`; `MYOSUITE_BASELINES_REVISION` or the `revision` argument override it), and fall back to `main` when no tag exists or the tag does not hold the file; they print the revision they used. Write the manifests of a baselines checkout with `python scripts/write_checkpoint_manifests.py <dir>`.
