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
| myoLegHillyTerrainWalk-v0    | `model_4999.pt`  | 15.3%³                        |
| myoLegRoughTerrainWalk-v0    | `model_4176.pt`  | 91.0%¹                        |
| myoLegStairTerrainWalk-v0    | -                  | -                             |
| myoLegStandRandom-v0         | `model_3500.pt`  | 34.1%                         |
| myoLegWalk-v0                | `model_3000.pt`  | 100.0%                        |
| myoTorsoExoPoseFixed-v0      | `model_188.pt`   | 100.0%                        |
| myoTorsoPoseFixed-v0         | `model_103.pt`   | 100.0%                        |

¹ Resumed on MyoSuite 3.0.0 from a run trained with mjlab 1.3.

² `myoLegDirectionalForward-v0` and `myoLegDirectionalBackward-v0` swapped names after 3.0.0: the model faces `-y`, so the task that
walks along `-y` is now the forward one. Each checkpoint moved with its task definition; a checkpoint downloaded under the old
name walks in the opposite direction of the new one.

³ Mean of seeds 1-3 with 256 episodes each. The earlier 26.6% was a single draw of 64 episodes; the twin bakes one terrain per seed,
so one draw varies a lot. The released 3.0.0 code gives 18.4, 19.9 and 14.1% on the same seeds, so the change is not a regression.

CPU evaluations that pass a NumPy array as `FatigueWrapper.fatigue_reset_vec` now reuse the requested initial fatigue in every episode. Earlier versions mutated that array during stepping, which changed later episodes. Reevaluate checkpoints measured with this configuration.

---

<sub>

**Not all published policies converged to the standard 95% threshold**, and training for more
iterations may well reach higher success rates: `myoHandPoseRandom-v0` (36%, plateaued),
`myoArmReachRandom-v0` (80%), `myoFingerReachRandom-v0` (73%),
`myoLegDirectionalRandom-v0` (54%), `myoLegStandRandom-v0` (34%) and `myoLegHillyTerrainWalk-v0`
(15%). `myoLegDirectionalRandom-v0` succeeds with exploration noise (99% sampled success) but only about half of the time with the mean
action; the checkpoint is the best one of the run.

</sub>

<sub>

Re-evaluate a checkpoint after any change to an env's observations with
`python scripts/eval_mjlab_policy.py <env_id> --backend mjlab`; train it again if its success rate drops
noticeably (with 64 episodes, differences of a few points are noise). Success rates were measured with
64 parallel episodes of the deterministic policy.

</sub>

<sub>

**Terrain and CPU evaluation.** A deterministic CPU evaluation is a single trajectory. The mjlab twins of the
rough, hilly and stairs walk tasks bake one terrain into the height field (Warp shares one `hfield_data` between
worlds), while the CPU envs draw a new terrain at every reset, so CPU and mjlab success rates on these ids
measure different things. Tiling several height-field patches in one field, one per env, would match the CPU
distribution; this is not implemented.

</sub>

<sub>

**Older policies and datasets.** Policies trained on the earlier CPU `myoChallengeChaseTagFBP2-v0` contract
(1496-dim observation, 0.02 s control step, random terrain) no longer load, and SAR activation datasets or synergy
models collected before the MuscleMimic bridge fix (#459) must be recollected.

</sub>

<sub>

**Collision bounds and inertia of randomized objects.** Die-reorient, Baoding P2 and weighted elbow-pose episodes now use refreshed collision bounds and mass-consistent inertia, so their dynamics can differ from earlier builds; re-evaluate policies and recollect affected datasets.

</sub>

<sub>

**Motor finger gears.** With `myo_sim` 0.2.4 the `motorFinger*` envs use the gears of `motorfinger_v0.xml` as they are, 160/80/80/160/160 (`extn`, `adabR`, `adabL`, `mflx`, `dflx`), and MyoSuite no longer scales them. Checkpoints trained on the earlier 80/20/20/40/40 no longer apply. Their observations, actions and control step are unchanged, so they still load and the env contract hash in `manifest.json` does not warn about them: use the checkpoints of the release that matches your MyoSuite version. The four `motorFinger*` rows are means over seeds 0-2 with 256 episodes each.

</sub>

<sub>

**Fatigue parameters of torso, hamstring, forearm and two-agent muscles.** In the `myoFati*` variants, lumbar and
abdominal muscles use the new `Trunk` row and hamstrings the `Thigh-Hamstring` row (both used `Default`), the forearm
rotators `SUP`/`PT`/`PQ` the `Elbow` row and `APL` the wrist-extensor row, and the left arm of the MuscleMimic models
and every muscle of two-agent scenes get their muscle group's row. The checkpoints above (base envs) are unchanged; fatigue results of these variants from earlier
builds are not comparable.

</sub>

<sub>

**MuscleMimic full body (single clip).** `checkpoints/myoMimicFullbody-v0-walking_medium06/model_81380.pt` on the
Hugging Face repo imitates the clip `KIT/167/walking_medium06` (AMASS / KIT motion capture retargeted with GMR; cite AMASS, KIT and MuscleMimic, see the *References* page; 2.0B steps, 1024 envs, about 30 h on one GPU). It
is scored by clip completion instead of success: 86% of 576 mean-action episodes from random start frames play to
the end of the clip (84-88% per run), the rest end on a pose deviation. It is an mjlab-only baseline (no CPU twin: the
CPU `myoMimicFullbody-v0` uses random targets), so it sits in its own folder and is loaded with `--checkpoint` and
`MIMIC_CLIP` (tutorial 5.3). It was trained on the clip's stored references: evaluate it with
`MYOSUITE_MIMIC_CLIP_REFERENCES=stored` (see *Clip references on the joint couplings* below).

</sub>

<sub>

**Coupled knee joints at reset (MuscleMimic models).** The CPU envs `myoMimicFullbody-v0`, `myoFullBodyWaypoint-v0`
and `myoFullBodyDirectional-v0` and their mjlab tasks now start with the tibia and patella joints of each knee on their
joint couplings. From `qpos0` and the keyframe the patella started up to 5.2 cm off them and was pulled back over the
first 0.3 s. Clip starts (RSI, the directional env) were further off: the retargeted clips keep the
`musclemimic_models` 1.0.5 sign of the left knee's `rotation3` and `translation2` couplings, up to 0.5 rad off on 1.0.6.
The observation, action, control step and reward definitions are unchanged, so the env contract hash in `manifest.json`
does not warn and the checkpoints still load. Only the first steps of an episode differ, but re-evaluate a policy before
comparing it with new runs. The single-clip policy above was trained with the unprojected clip starts; its clip
completion with projected starts is in the next note.

</sub>

<sub>

**Clip references on the joint couplings (MuscleMimic clips).** The retargeted clips keep the left knee's `rotation3`
on the `musclemimic_models` 1.0.5 polynomial, up to 0.50 rad and 3.8 rad/s off the 1.0.6 coupling, and their site
targets are the kinematics of that pose. The clip tasks (mjlab `myoMimicFullbody-v0` / `myoMimicBimanual-v0` with a
clip, `MuscleMimicClipEnvV0`, the gait reward of `myoFullBodyDirectional-v0`) now serve every frame on the couplings of
the model in use, with the site targets moved with the joints: on `walking_medium06` the left ankle and toe targets
move by up to 33 and 44 mm (mean 8.5 and 10.5 mm). The observation and reward definitions are unchanged but their
values are not, and the env contract hash does not warn. `MYOSUITE_MIMIC_CLIP_REFERENCES=stored` serves the clips as
stored (episodes still start on the couplings), for checkpoints trained on them. The single-clip policy above is one:
its observation normalizer puts the corrected `rotation3` reference and velocity up to 6.2 and 6.7 standard
deviations from the stored ones it was trained on, and it completes fewer clips on them. Clip completion of
`model_81380.pt` (mjlab, mean action, random start frames, 3 seeds x 64 envs x 3 episodes = 576 episodes per row,
measured on an RTX 4090 with mjlab 1.6.0 and MuJoCo Warp 3.11):

| Build | Clip completion [95% CI] | Per seed |
|---|---|---|
| Release 3.0.0 | 82.5% [79.1, 85.4] | 83.3, 82.8, 81.2 |
| Before the coupling fixes | 81.4% [78.0, 84.4] | 83.9, 80.7, 79.7 |
| Starts on the couplings, references as stored (`MYOSUITE_MIMIC_CLIP_REFERENCES=stored`) | 83.9% [80.6, 86.6] | 84.4, 82.8, 84.4 |
| Starts and references on the couplings (default) | 73.3% [69.5, 76.7] | 72.9, 75.0, 71.9 |

The first three rows do not differ (paired on start frames, McNemar p >= 0.4); the default is lower than release
3.0.0 (p = 0.005), most of all for start frames 100-300. The 86% above was measured at release 3.0.0 with another
software stack; this stack gives 82.5% for the same release. A policy trained on the corrected references needs a new
run.

</sub>

<sub>

**Bounded action std in the PPO defaults (#566).** The mjlab PPO configs of the muscle tasks and of `motorFinger*`
(whose actions are clipped before the linear map to `ctrlrange`) now keep the policy's action std in [0.05, 1.0]. The
std parameter is projected onto that range before every forward pass and stays trainable at the bounds. MuscleMimic
(with the SAR tasks of tutorial 5.5, which register with its config), ChaseTag FBP2, table tennis and the elbow config
also start at std 0.5 and use an entropy bonus of 0.001 (were 1.0 and 0.01). The bound limits the drift of the std of
saturated controls (up to the learning rate per update, less with the noise of PPO's surrogate gradient); it does not
by itself prevent saturated (bang-bang) controls. The checkpoints above were trained with the earlier, unbounded
defaults. They still load, and `eval_mjlab_policy.py` scores them as before, with or without `--stochastic`: the bound
adds no parameter, and a run whose `params/agent.yaml` has no `std_range` is sampled with the checkpoint's own std
(runs trained with the bound are sampled with the std clamped to their range, as in training). Resuming training from
one of them projects a std above 1 to 1 on the first update (the MuscleMimic run of #566 ended at 14), and the std stays
trainable. A run with the new defaults is a new training setup, so its policy and success rate can differ from the
published ones. This includes pose, reach, leg, torso and `motorFinger*`, whose std was unbounded before (an elbow-pose
run without the bound took its smallest std below 0.05 after 220 iterations). The run times and results quoted in
tutorial 5.5 were measured with the earlier defaults.

</sub>

<sub>

**`myoChallengeChaseTagFBP2-v0`** (success: the agent tags the opponent) and
**`myoChallengeTableTennisP{0,1,2}-v0`** have no trained checkpoint available yet to evaluate.

</sub>

<sub>

**`myoChallengeChaseTagFBVs-v0`** (the scored 1v1 game, CPU only). `onnx/chasetag_fbvs_baseline.onnx` in the `myohub/myosuite-3-baselines` Hugging Face repo at revision `ed40fb4f0a53` (`myosuite.core.hf_io.download_baseline_file`) plays one player, 546 -> 354, in either role. It is a starting point: it reaches and tags a player that does nothing, but it often falls before the 10 s limit: at about 6.7 s as a chaser against itself. Measured with action noise 0.02 and 12 episodes per role, on myo_sim 0.2.4 and MuJoCo 3.11; myo_sim 0.2.3 and MuJoCo 3.17 gave the same tags and falls:

| Match | Tags | Chaser falls | Mean length | Fell at (median) |
|---|---:|---:|---:|---|
| Chases a player that does nothing, as player A | 12/12 | 0 | 3.0 s | - |
| Chases a player that does nothing, as player B | 12/12 | 0 | 3.6 s | - |
| Against itself, A chases | 11/12 | 0 | 5.2 s | runner 1.7 s |
| Against itself, B chases | 0/12 | 12 | 7.1 s | runner 5.5 s, chaser 6.7 s |

It was fine-tuned with self-play on `myofullbody` (4,000 iterations, 2048 envs) from a policy that descends from the behaviour-cloned tutorial 5.4 walker (`policy_bc_best.pt`, a student of MuscleMimic `amathislab/mm-10m-2`), so it is not trained from scratch on this body. Observations and actions match the game; only the dynamics of `myofullbody` were new to it. Policies trained on `musclemimic_models` do not carry over reliably: moving between releases 1.0.6 and 1.0.5 of that body already took a similar policy to 0 tags.

</sub>

<sub>

**Manifests and releases.** Every `checkpoints/<env_id>/` folder on the Hugging Face repo has a `manifest.json` with the env id, a hash of the env contract (observation and action shapes, control step), the MyoSuite version and commit, and the success rate above. When a downloaded checkpoint is used, MyoSuite compares that hash with the current env and warns if they differ, because the policy may then not work. The repo is tagged per release (`v3.0`, `v3.1`, ...), and per patch release where baselines changed (`v3.0.1`); `download_baseline_checkpoint` (one env) and `download_baseline_file` (a single file, such as the MuscleMimic policy) take the tag of your installed version, the patch tag first (`v3.0.1`, then `v3.0`; `MYOSUITE_BASELINES_REVISION` or the `revision` argument override it), and fall back to `main` when no tag exists; they print the revision they used. Write the manifests of a baselines checkout with `python scripts/write_checkpoint_manifests.py <dir>`.

</sub>
