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

A body scaled to a subject (`make_env(env_id, scales=..., total_mass=...)`) changes the
dynamics even though the observation and action shapes stay the same. The published
checkpoints are trained on the nominal body. A policy trained on a subject should record it
with `write_manifest(..., scales=..., total_mass=...)` and be checked with
`check_contract(..., scales=..., total_mass=..., strict=True)` before loading. A different
or unrecorded body is then rejected. Nominal checkpoints are checked as before. See the
body-scaling guide.

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
| myoLegHillyTerrainWalk-v0    | `model_4999.pt`  | 0%³ ⁵ (20.7% with random hill heights) |
| myoLegRoughTerrainWalk-v0    | `model_4176.pt`  | 91.0%¹ ⁵                      |
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

⁴ **Incompatible with MyoSuite 3.1.0.** Measured before 3.1 from the old start. From 3.1 the leg tasks reset with the coupled knee joints on their polynomials (#630), and this checkpoint falls from that start (0% success, *Coupled knee joints at reset (leg tasks)* below). A checkpoint trained from the new start follows in 3.2. It was also trained with the earlier `com_vel` (footnote ⁵).

⁵ Trained and measured before the `com_vel` fix (*Centre-of-mass velocity of the walk tasks* below) and not re-evaluated since.

CPU evaluations that pass a NumPy array as `FatigueWrapper.fatigue_reset_vec` now reuse the requested initial fatigue in every episode. Earlier versions mutated that array during stepping, which changed later episodes. Reevaluate checkpoints measured with this configuration.

`myoChallengeChaseTagP2-v0`, `myoChallengeChaseTagP2eval-v0` and their `myoSarc` / `myoFati` variants pay the EVADE win again: the last step of an EVADE episode that reaches the time limit has `solved = 1` (+1000) and score 1.0. In 3.1.0 that win could not fire, so EVADE returns and `get_metrics` scores measured on 3.1.0 are too low for policies that survive; reevaluate them. Observations, actions and dynamics are unchanged, and no checkpoint of these ids is published here.

On every CPU env, a step on which MuJoCo reset a diverged simulation no longer pays a positive reward or counts as `solved`, and `info["mj_unstable"]` marks it. Observations, actions, dynamics and the rewards of regular steps are unchanged, so checkpoints load and score as before; only evaluations that contained a diverged episode change (the ChaseTag envs counted such an episode as a win).

The two-agent `myoChallengeChaseTagFBVs-v0` now ends such an episode for both players with `info["score"] == 0.0` (it continued from the reset state and could count it as a tag or an evade timeout win); evaluators should treat `info["mj_unstable"] == True` as a void bout (draw, score 0). Its baseline `onnx/chasetag_fbvs_baseline.onnx` is unaffected outside diverged episodes.

`myoMimic-v0` uniform and return resets now exclude terminal-only reference bins on CPU and mjlab, so every reset has at least one reference transition. This corrects the random-start distribution for clips whose last frame starts a bin (for example, a one-second clip at 10 ms). The stored terminal frame remains the final scoring target. Reevaluate policies with the corrected distribution; adaptive per-bin training statistics from a library containing those empty bins must be rebuilt. Policy input/action layouts and clip-start evaluation are unchanged.
CPU clip-following evaluations through `MuscleMimicClipEnvV0` now score sites after the final physics substep. Earlier site-based rewards and early termination used geometry from one substep earlier than qpos/qvel. Reevaluate checkpoints measured through this class: policy inputs, action layout, control step and integration trajectories are preserved, while rewards and termination can differ.
`myoChallengeBimanual-v0` and its `myoSarc` / `myoFati` variants draw the box size on the first `reset()` of an env instance too; before, the first episode of every instance used the nominal box. An evaluation that builds a fresh env per episode now sees the size randomisation (10 %, 5 %, 10 % per axis), and a seeded first episode differs from 3.1.0. `get_metrics` counts a path with at least one solved step as a success (before: more than 5, which no episode could reach, so the score was always 0). Observations, actions and rewards are unchanged, and no checkpoint of these ids is published here.
`myoFullBodyWalkSAR-v0` (mjlab only, tutorial 5.5) now rewards the root velocity along the body's heading. Up to 3.1.0 it rewarded world `+y`, which is backwards for the MuscleMimic full body, so a policy trained before this change walks backwards: retrain it. Observations, actions and dynamics are unchanged, so such a policy still loads. No checkpoint of this id is published here.
`myoMimic-v0` (mjlab twin): with `reset.auto_reset` (the default) the return-based reset statistics now include the last step of every episode, so starts near a clip end are no longer over-sampled and the reset distribution during training differs from 3.1.0. Observations, actions, rewards and the CPU env are unchanged. No `myoMimic-v0` checkpoint is published here, and the myoSuperMimic tutorial files were trained with resets driven by the runner (`reset.auto_reset=False`), which this does not change.
In `myoChallengeChaseTagP2-v0`, `myoChallengeChaseTagP2eval-v0` and their variants, the `opponent_vel` observation of EVADE episodes is the opponent's applied velocity (capped at 2 m/s and 2 rad/s) instead of its uncapped command. Dynamics and rewards are unchanged, but a policy trained on the old values sees different inputs in EVADE episodes; no checkpoint of these ids is published here.
`myoChallengeChaseTagFBP2-v0` (CPU and mjlab): the `rel_vel` slots of the `opponent_relative` observation block (indices 3-5 of the block) now hold the opponent's world-frame velocity minus the agent's, instead of the opponent's `[speed, turn rate, 0]` command minus the agent's velocity. Policies trained on the earlier FBP2 observation see different values in these three slots and should be fine-tuned or retrained; no FBP2 checkpoint is published here.
`myoChallengeSoccerP1/P2-v0` and `myoChallengeBimanual-v0` (and their variants) end a time-out with `truncated=True, terminated=False`; Soccer episodes last 1000 steps (was 1001) and its time-out step pays `alive` instead of counting in `sparse`. Observations and the rewards of all other steps are unchanged; no checkpoint of these ids is published here.
`myoChallengeDieReorientDemo/P1/P2-v0` and `myoChallengeRelocateP1/P2/P2eval-v0` (and their variants) compute the rotation error as the angle between the object and the goal, not as the norm of Euler-angle differences. Die Reorient success rates rise for policies that end near the goal (a die 0.05 rad off its goal was counted as failed for 9 % of the P2 goals), and the `rot_dist` reward changes on all six ids. Observations are unchanged, so checkpoints still load; reevaluate success rates and retrain reward-sensitive comparisons. No checkpoint of these ids is published here.
SAR tasks (`myoMimicFullbody-SAR-v0`, `myoMimicBimanual-SAR-v0`, their `myoMuscleMimic*` aliases and `myoFullBodyWalkSAR-v0`; mjlab only, tutorial 5.5): a policy action in `[-1, 1]` now spans the fitted synergy range. Up to 3.1.0 the synergies were scaled to `[0, 1]`, so action 0 was the lower edge of every synergy and negative actions lay below the data. Observation and action shapes are unchanged, so a SAR policy trained before this change still loads, but it decodes to other muscle activations: retrain it. Saved synergy models are rescaled when loaded and need no re-extraction. No SAR policy or synergy model is published here.
The Die Reorient ids (`myoChallengeDieReorientDemo/P1/P2-v0` and variants) place the die geoms from their compiled positions at every reset, so a seeded episode no longer depends on the previous one. Positions move by at most 1e-16 and trajectories by about 1e-13: no checkpoint needs reevaluation.
Pure `joint_penalty` and `joint_limit_violation` terms now read `EnvAccessor.joint_limits()` (`jnt_range` on limited 1-DoF hinge/slide coordinates) rather than `actuator_ctrlrange`, `MuscleActionTerm` counts only the entity's muscle actuators when `muscle_fatigue=True`, and `ChaseTagVs` `compute_damage` / `get_score` apply tag KO damage to the active runner in `EVADE` bouts and clamp scores to `[0, 1]` (#803). None of the published base-env checkpoints in the table above use `joint_penalty`, `joint_limit_violation`, or `MuscleActionTerm(muscle_fatigue=True)`, so their observations, actions and deterministic rollouts are unchanged; re-evaluate custom `TaskConfig` / mjlab runs or `EVADE` multi-agent matches that relied on the old behaviour.
`ChaseTagField.flatten_agent_patch` now flattens the terrain patch under `(y, x)` rather than `(x, y)` on non-flat `myoChallengeChaseTagP2*` episodes, and `ChaseTagEnv` seeds `self.np_random` before initialising terrain/opponents and clamps keyframe indices when `nkey == 1`. Observations, actions and flat-arena episodes are unchanged; no checkpoint of these ids is published here.
`myoChallengeBimanual-v0` and its variants now advance `touch_history`, `max_force` and `goal_touch` once per `reset()` / `step()` instead of mutating episode state when `get_obs_dict()` or `get_reward_dict()` is inspected externally. Standard `reset()`/`step()` trajectories and observations are unchanged.
`myoChallengeTableTennisP{0,1,2}-v0` and variants now run `mj_forward` and refresh `obs_dict` immediately after a multi-rally `relaunch_ball()` (`rally_count > 1`), sample `cal_ball_qvel` when `ball_qvel=True` even if `ball_xyz_range` is `None`, and keep `get_obs_dict()` / `get_reward_dict()` side-effect-free. Single-rally P0/P1 rollouts are unchanged; no checkpoint of these ids is published here.
`myoChallengeSoccerP1/P2-v0` and variants now report `rwd_dict["sparse"] = float(goal_scored)` instead of `float(done)` (which previously paid sparse points on a fall) and enforce `min_agent_spawn_distance` at reset when positive. Dense rewards (`sparse` weight is 0) and observations at default spawn settings are unchanged; no checkpoint of these ids is published here.
CPU `myoLegStandRandom-v0` (`LegReachEnvV0._generate_qpos`) now perturbs and clips only `jnt_limited` joints instead of clamping the pelvis freejoint `qpos[0]` to `0.0` at reset. Observation/action shapes and checkpoint loading are unchanged; CPU reset evaluations keep the keyframe pelvis X position.

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
  `onnx/chasetag_fbvs_baseline.onnx` (*`myoChallengeChaseTagFBVs-v0`* below). After 3.1.0 its training reward
  changed (potential-based proximity, no per-step survival reward for the chaser); observations, actions and the
  score did not, so the baseline still plays, but it was trained on the old reward.
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

**Success of the walk tasks (after 3.1.0).** `solved` of `myoLegWalk-v0`, the rough, hilly and stairs walks and their
`myoSarc`/`myoFati` variants, and `Episode_Metrics/success` of their twins, now mean sustained walking: upright and
within 0.5 m/s of the target velocity on every control step of the last second. Up to 3.1.0 one such step was enough,
and the reset keyframe already moves at about the target speed: a zero-action episode that falls after about 0.2 s had
7 to 10 solved steps, and `evaluate_success` (more than 5 solved steps) gave 100 %. The rates in the table are measured on
the last step of an episode. An episode that ends in a fall failed before and fails now; an episode that reaches the
time limit now also needs its whole last second on target. The walk rows were measured with the earlier flag and not
re-evaluated: they can only stay or drop. Rates from `evaluate_success`, or from any rule that counts an episode with
one solved step, are not comparable with earlier versions. Observations, actions, rewards and dynamics are unchanged,
so no checkpoint needs retraining. The per-step flag is still reported as `on_target` (CPU `info`) and
`Episode_Metrics/on_target` (mjlab, share of an episode's steps).

The directional walks (`myoLegDirectionalForward-v0`, `myoLegDirectionalBackward-v0`, `myoLegDirectionalRandom-v0`)
use the same rule (one second is 200 of their 0.005 s control steps, of 500 per episode). Random actions that fall
after about 1 s had 15 to 69 solved steps and scored 100 % with `evaluate_success`; they now score 0 %. Their rows
(96.9 %, 100 % and 53.8 %) were measured on the last step with the per-step flag and not re-evaluated: they can only
stay or drop.

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

**Passive force of the sarcopenia variants (after 3.1.0).** The `myoSarc*` ids, their mjlab twins and the
`elbow_sarcopenia` recipe now scale the passive muscle force with the peak force: `biasprm[2]` of every muscle bias
gets the same factor (0.5) as `gainprm[2]`, with automatic peak forces (`force="-1"`) resolved first. Up to 3.1.0 the
passive force stayed at full strength (ratio 1.000 on every id), as in 2.x. The table lists no `myoSarc*`
checkpoint (they use the checkpoint of their base env, whose dynamics are unchanged), but results and policies on
the sarcopenia variants from earlier builds are not comparable: passive joint torques are now half as large.

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

**Random-target MuscleMimic tasks (after 3.1.0).** Without a clip, `myoMimicFullbody-v0`, `myoMimicBimanual-v0` and
their aliases `myoMuscleMimicFullbody-v0` / `myoMuscleMimicBimanual-v0` (CPU and mjlab) now draw each episode's
targets as the tracked sites of one random posture: joint angles uniform within their limits, coupled joints on their
couplings and within their ranges, the full body resting on the floor its standing pose stands on. Before, every site
got an independent target in one box, which broke the segment lengths (and moved the immovable torso site of the
fixed-base bimanual model), so no policy could reach `solved` (mean site error under 0.04 m). The observation, action
and reward definitions are unchanged, but the target distribution is not: policies trained on the box targets were
trained on another task. No random-target checkpoint is published; the single-clip checkpoint above tracks its clip
and is unaffected.

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

**Coupled joints at reset (MyoChallenge tasks, after 3.1.0).** The CPU envs `myoChallengeChaseTagP1/P2/P2eval-v0`,
`myoChallengeChaseTagFBP2-v0` (and its mjlab twin), `myoChallengeOslRunFixed/Random-v0`,
`myoChallengeRelocateP1/P2/P2eval-v0`, `myoChallengeBimanual-v0`, `myoChallengeSoccerP1/P2-v0` and their `myoFati` /
`myoSarc` variants now also start with the coupled joints on their joint couplings; 3.1.0 had left them as they were.
The start state moves most for ChaseTag P1/P2/P2eval (walk keyframes 2 and 3 of `myoLegWalk-v0`: patella rotation
0.74 rad, feet up to 1.6 cm), Relocate (shoulder rhythm up to 0.16 rad, palm 4.3 cm in P1 and 3.0 cm in P2) and OslRun
(0.16 rad, biological foot up to 1.4 cm); Bimanual, Soccer P2 and ChaseTag FBP2 move by 2.5 mm, 2.3 cm (trunk, from
the reset noise) and 0.4 mm. The observation, action, control step and reward definitions are unchanged, so the
env contract hash does not warn. No checkpoint of these ids is published in `myohub/myosuite-3-baselines` (the table
above has none for `myoChallengeChaseTagFBP2-v0`), so none was re-evaluated. Policies trained on these ids with
MyoSuite 3.1.0 or earlier, or with the PyPI 2.x MyoChallenge envs, were trained from the old starts: re-evaluate them
before comparing, in particular ChaseTag P1/P2 policies, which share the keyframe from whose projected start the
`myoLegWalk-v0` checkpoint falls. `myoChallengeTableTennisP{0,1,2}-v0` (and their mjlab twins) start from a
re-solved right-arm pose on the couplings: the seven arm joints move by up to 0.42 rad and the shoulder rhythm by up to
0.59 rad, while the hand keeps its grasp on the paddle (grasp site within 5e-6 m and 2e-4 rad, same contacts); no
table-tennis checkpoint is published, but policies trained on the old keyframe start from a different arm pose.
`myoChallengeChaseTagFBVs-v0` is unchanged, and the tutorial files below were measured on its two-agent reset.

**Centre-of-mass velocity of the walk tasks (after 3.1.0).** `myoLegWalk-v0`, the rough, hilly and stairs walks and
their `myoSarc`/`myoFati` variants now observe and reward the velocity of the whole-body centre of mass, on both
backends. Up to 3.1.0 `com_vel` was the mass-weighted mean of the linear part of MuJoCo's `cvel`, which leaves out the
rotation of the segments: 0.20 to 0.27 m/s off on average and up to 1.06 m/s along the walking direction on
random-action rollouts, and 1.30 instead of 1.76 m/s at the reset keyframe. Two of the 403 observation values, the
velocity reward and the `solved` flag change; shapes, keys, actions, control step and dynamics do not, so the env
contract hash in `manifest.json` does not warn and the checkpoints still load. Affected published checkpoints:
`myoLegWalk-v0` `model_3000.pt` (already incompatible with 3.1.0), `myoLegRoughTerrainWalk-v0` `model_4176.pt` and
`myoLegHillyTerrainWalk-v0` `model_4999.pt`; the stairs walk has no checkpoint and the `myoSarc`/`myoFati` variants use
the checkpoint of their base env. They were trained on the earlier value, as an input and as the quantity the
1.2 m/s target applies to, and were not re-evaluated after the fix: re-evaluate them, and retrain before comparing
walking speeds or success rates with new runs.

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
