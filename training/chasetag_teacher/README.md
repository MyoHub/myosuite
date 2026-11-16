# ChaseTag teacher: a steered myoSuperMimic tracker and its DAgger student (experimental)

A teacher player for the scored 1v1 game `myoChallengeChaseTagFBVs-v0`, built from a motion-tracking policy instead
of a walking policy, and its distillation into one 546 -> 354 network that plays both roles:

1. **Teacher.** A myoSuperMimic tracker (`training/supermimic`, trained on `myoMimic-v0` with myo_sim's
   `myofullbody`, the body of the FBVs players) plays one player on the CPU. A planner asks for a travel heading,
   toward the opponent (chaser) or away from it (runner) and turned off the arena fence
   (`myosuite.integrations.musclemimic.steering.aim_heading_xy`), and a speed. A steered reference turns that
   request into the tracker's goal: it plays the KIT locomotion clips of the tracker's training split under a yaw
   transform, jumps between clips by motion matching (pose and future-trajectory features) and warps the yaw within
   limits for the heading error left after the clip's own turning.
2. **Student.** DAgger distils the teacher into an MLP on the player's own 546-d FBVs observation (the
   `ActorCritic` of `myosuite.envs.myo.tasks.mimic.policy`, 512 x 6). Each round plays bouts in which the labelled
   players execute the previous student's action with probability 1 - beta (the teacher reads the executed
   excitation as its previous action, and in half the bouts takes over when the student's pelvis drops), against a
   second teacher, the shipped baseline or an idle player, from either side; the student is refitted (MSE on the
   excitations) on the data of all rounds.
3. **Yaw canonicalisation.** Rotating the whole scene about the vertical axis is an exact symmetry of the
   observation (checked against the simulator in `tests/test_scene.py`). The student reads the observation rotated by
   minus the player's own root yaw (`distill.canonicalise`), so it is invariant to that rotation by construction,
   instead of having to learn it from augmented data (random-yaw augmentation made the closed-loop student fall).
   The ONNX export puts the canonicalisation in the graph (`distill.export_canon_onnx`), so the file still takes the
   raw 546-d observation and outputs 354 excitations in [0, 1].
4. **Evaluation.** `scripts/evaluate.py` plays every match of a matrix of players (teacher, ONNX students, the
   shipped baseline, an idle player) from both sides and reports tags and wins per player.
5. **Directional walker** (`directional`, tutorial 5.4). The same teacher, asked for a world heading instead of a role,
   walks one `myofullbody` on flat ground; DAgger (`scripts/dagger_directional.py`) distils it into a 528 -> 354
   walker. The walker reads its joints, velocities, activations, root velocity in the yaw frame and the commanded
   heading **in the yaw frame**, so it is invariant to rotations about the vertical axis by construction.
   `export_chasetag_onnx` puts a 546 -> 528 adapter in front of it (heading toward or away from the opponent by the
   role flag, turned off the fence like the teacher's planner) and writes a scored-game player (tutorial 5.6).

This is a self-contained training project (`training/chasetag_teacher/`, import name `chasetag_teacher`), not
part of the installable `myosuite` package; see `docs/wiki/training-guidelines.md`. MyoSuite never imports it.

It replaces the earlier content of this project, a writer of (observation, action) pairs from the `mm-10m-2`
steered teacher. That teacher stays available for playback as `ChaseTagSteeredPair` in
`myosuite.integrations.musclemimic.steered_teacher`. Tutorial 5.7 plays the myoSuperMimic teacher of this project.

## Attribution

The code in this project is new MyoSuite code (header `Copyright (c) MyoSuite Authors`). It builds on:

| Work | Used for |
|---|---|
| SuperMimic, [Liao et al., arXiv:2610.28480](https://arxiv.org/abs/2610.28480), through MyoSuite's port `training/supermimic` (attributions and licences there) | the tracker and its policy network, kinematics and checkpoints |
| DAgger, Ross, Gordon and Bagnell, AISTATS 2011 ([arXiv:1011.0686](https://arxiv.org/abs/1011.0686)) | the distillation loop |
| Motion matching, Clavet, GDC 2016 | the clip transitions of the steered reference (reimplemented here) |
| [AMASS](https://amass.is.tue.mpg.de) (Mahmood et al., ICCV 2019) and the KIT whole-body human motion database (Mandery et al., 2015, 2016); MuscleMimic's KIT split (Li et al., [arXiv:2603.25544](https://arxiv.org/abs/2603.25544), from KINESIS) | the motions of the steering library |

## Data and checkpoints

No motion clip is shipped. Tutorials 5.4, 5.6 and 5.7 download the 30-minute tracker
(`supermimic/tracker_myofullbody_kit_30min.pt`), the directional walker (`directional/walker_myofullbody.pt`) and the
ChaseTag student (`onnx/chasetag_supermimic_student.onnx`, and a self-play refinement of it,
`onnx/chasetag_supermimic_selfplay.onnx`) from the baselines repo
(`myosuite.core.hf_io.download_baseline_file`).

- **Clips.** Retarget the AMASS KIT motions from your own AMASS download onto `myofullbody` with
  `scripts/mimic/retarget_amass.py` (in MyoSuite) and point `$MYOSUITE_KIT_ROOT` at the output folder, or name a
  Hugging Face dataset of retargeted clips in `$MYOSUITE_MIMIC_DATASET` (`myosuite.envs.myo.tasks.mimic.tracking.kit_root`).
  The AMASS licence (non-commercial research) applies to the clips and to anything derived from them.
- **Tracker.** Train one with `training/supermimic` (its fast recipe; a 30-minute run was the teacher of the results
  below). The tracker must be trained on the same myo_sim version as the FBVs body (`check_same_body`).

The motion data of the 860 locomotion clips is built once and cached in `$CHASETAG_TEACHER_CACHE` (default
`~/.cache/myosuite/chasetag_teacher`, about 100 MB) as NumPy arrays without pickles. The cache is keyed by the bytes
of every clip, the body, the control step and the preprocessing code, so re-retargeted clips rebuild it; `lib_*.pt`
files of earlier versions are no longer read and can be deleted.

## Install

```bash
pip install "myosuite[mjlab]"
pip install "git+https://github.com/MyoHub/myosuite.git@v3.1.0#subdirectory=training/supermimic"
pip install "git+https://github.com/MyoHub/myosuite.git@v3.1.0#subdirectory=training/chasetag_teacher"
# or, from a checkout: pip install -e ".[mjlab]" -e training/supermimic -e training/chasetag_teacher
```

Install all three from the same branch.

## Use

```bash
# the steering library (once)
python training/chasetag_teacher/scripts/build_library.py
# port checks: the FBVs players have the tracker's body; the CPU tracker's observation and goal match myoMimic-v0;
# the tracker follows held-out clips
python training/chasetag_teacher/scripts/check_port.py <tracker>/model_014164.pt --clips 6
# DAgger as for the results below: a plain student for 20 rounds, then the canonicalised one, fitted fresh at round 20
python training/chasetag_teacher/scripts/dagger.py --out runs/dagger --checkpoint <tracker>/model_014164.pt \
    --rounds 22 --bouts 480 --betas 1,0.3,0.1,0 --takeover-z 0.8 --canon --canon-from 20
# refit a student offline on the collected data
python training/chasetag_teacher/scripts/retrain.py --out runs/s.pt --data runs/dagger/data/r* --canon
# the evaluation matrix: every match from both sides, 48 episodes each
python training/chasetag_teacher/scripts/evaluate.py --out runs/eval.json --sm teacher=<tracker>/model_014164.pt \
    --onnx s21=runs/dagger/student_r21.onnx \
    --matches teacher:idle,teacher:base,base:teacher,s21:idle,s21:base,base:s21 --episodes 48
# the directional walker (tutorial 5.4): DAgger with tutorial 5.4's evaluation after every round
python training/chasetag_teacher/scripts/dagger_directional.py --out runs/walker --checkpoint <tracker>/model_014164.pt \
    --rounds 8 --episodes 480 --workers 12
```

`student_r<k>.onnx` is a scored-game player: `obs [batch, 546] -> action [batch, 354]`. In Python,
`match.SmPlayer` (teacher), `distill.StudentPlayer` and `match.OnnxPlayer` play bouts with `match.play`.

## Results (preliminary)

48 episodes per match and side, against the shipped FBVs baseline (`onnx/chasetag_fbvs_baseline.onnx`, see
`docs/baseline_checkpoints.md`) and an idle player (zero excitation). *Tags idle*: bouts chasing an idle player that
end in a tag. *Head-to-head*: bouts against the baseline won (a tag as chaser, no tag as runner).

| Player | Tags idle | Head-to-head vs shipped baseline |
|---|---:|---:|
| baseline | 100 % | 50 % |
| steered teacher (30-minute tracker + steering) | 92 % | 71 % |
| student, round 21 (yaw-canonicalised) | 98 % | 61 % |
| directional walker, round 5 (`scripts/dagger_directional.py`, as a player through `export_chasetag_onnx`) | 100 % | 64 % |

One tracker, one DAgger run and one seed per row; the earlier rounds trained a plain student on raw observations
and the canonicalised one started at round 20. These numbers will move with more runs.

The directional walker (12 rounds of 480 episodes; round 5 scored best) survives 91 % of tutorial 5.4's 2 s rollouts
(8 headings x 10 seeds from standing, alignment +0.85) and 85 % over 10 s (alignment +0.93); later rounds were not
better.

## What lives where

| | |
|---|---|
| MyoSuite | the scored game, the role and arena headings (`myosuite.integrations.musclemimic.steering`), the student network (`ActorCritic`), `export_fbvs_onnx`, the `myoMimic-v0` task, its motion library and the retargeting script |
| `training/supermimic` | the tracker: its training, checkpoints, kinematics (`SkeletonKinematics`, `rel_sites`) and policy network |
| this project | `library` / `reference` (steering library and reference), `tracker` / `player` (the tracker on one FBVs player), `match` (bouts, scores), `distill` / `augment` (DAgger, canonicalisation, ONNX), `directional` (the directional walker: scene, observation, teacher, evaluation, ChaseTag export) |

## Tests

```bash
cd training/chasetag_teacher && pytest
```

Without data the tests check the canonicalisation (exact invariance), the half-turn symmetry, the 546 -> 354
interface, ONNX against torch, the scene symmetry, the directional observation (layout, yaw invariance), its
ChaseTag adapter against the teacher's planner heading and its ONNX, the players' body against the tracker's, and the tracker's
observation and the reference goal against `myoMimic-v0` on synthetic clips. With `$MYOSUITE_KIT_ROOT` and
`$CHASETAG_TEACHER_CHECKPOINT` (a tracker `model_*.pt`) the `data` tests also play bouts; without them they skip.
MyoSuite's CI does not run these tests.
