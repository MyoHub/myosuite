# One ChaseTag student

One heading-conditioned walk policy, cloned from `mm-10m-2`. Same
weights for directional control and for scored ChaseTag 1v1.

The competition is two independent players. Each team submits **one**
ONNX: scored `myoChallengeChaseTagFBVs-v0` obs **546 → 354** muscles in
`[0, 1]`. Role lives in the observation (`[1, 0]` chase / `[0, 1]`
evade), not a second file.

The student itself is 528-d (`myoFullBodyDirectional-v0`). A thin
adapter rebuilds that vector from the scored 546-d obs and points
`heading_cmd` toward the opponent (or away), then clamps it to the
current facing. A 180° evade command from the standoff is out of
distribution for this walk student and falls. That is not a second policy.

Do not change the scored 546-d observation.

## Split

| PR | Owns |
|---|---|
| [#579](https://github.com/MyoHub/myosuite/pull/579) | 1.0.6 `HeadingTablePolicy` workaround (use only if forced onto 1.0.6) |
| [#578](https://github.com/MyoHub/myosuite/pull/578) | `myoChallengeChaseTagFBVsTrain-v0` self-play |
| [#587](https://github.com/MyoHub/myosuite/pull/587) | `ChaseTagSteeredPair` teacher + this student package |

Scored ChaseTag / FBVs pins **`musclemimic_models` 1.0.5** (`mm-10m-2` / 5.4 walker). Set `MYOSUITE_MUSCLEMIMIC_MODELS_VERSION=1.0.5` for local rolls.

## Install

```bash
pip install -e ".[musclemimic]"
pip install -e training/chasetag_teacher
```

## Train or adopt

Behavior-clone from 5.4 pairs (`obs (N, 528)`, `actions (N, 354)`):

```bash
python -m chasetag_teacher.train \
  --pairs /path/to/bc_directional.npz \
  --out runs/chasetag_student.pt
```

This Mac cannot roll the Orbax teacher. Adopt the already-trained IL
checkpoint from tutorial 5.6 (same 528 → 354 stack):

```bash
python -m chasetag_teacher.train \
  --from-ckpt ../../tutorials/files/5.6/policy_bc_best.pt \
  --out runs/chasetag_student.pt
```

Export the submission ONNX (same weights + 546→528 heading adapter):

```bash
python -m chasetag_teacher.fbvs_actor \
  --ckpt runs/chasetag_student.pt \
  --onnx runs/chasetag_student.onnx
```

## Two demos

Directional control — one student, eight headings:

```bash
python -m chasetag_teacher.play directional \
  --ckpt runs/chasetag_student.pt \
  --out runs/directional.mp4
```

1v1 — the same student loaded once, queried for both agents.
Spawn writes walk-clip frame 0 facing each other (same as directional
eval); after that only ``env.step`` moves them.

Also add a unit test for heading still. Run diagnostic + tests + export + render.

```bash
python -m chasetag_teacher.play versus \
  --onnx runs/chasetag_student.onnx \
  --out runs/versus.mp4
```

This is a walk prior (chase = walk toward, evade = walk away), not a
trained tag specialist. Further self-play is #578.

Teacher pair collection (546-d expert data, not this student) remains:

```bash
MYOSUITE_MUSCLEMIMIC_MODELS_VERSION=1.0.5 \
  python -m chasetag_teacher.collect --episodes 4 --steps 400 --out pairs.npz
```
