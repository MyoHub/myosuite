# ChaseTag teacher dataset

Collects `(546-d obs, 354-d action)` pairs from
`ChaseTagSteeredPair` on scored `myoChallengeChaseTagFBVs-v0`.
The pairs are the submission contract. Train a student with the
existing 5.4 / 5.6 / FBVs-train stack — do not add a second trainer here.

## Split

| PR | Owns |
|---|---|
| [#579](https://github.com/MyoHub/myosuite/pull/579) | 1.0.6 `HeadingTablePolicy`, tutorials 5.4 / 5.6 walker |
| [#578](https://github.com/MyoHub/myosuite/pull/578) | `myoChallengeChaseTagFBVsTrain-v0` (CPU + mjlab), self-play |
| [#584](https://github.com/MyoHub/myosuite/pull/584) ([#566](https://github.com/MyoHub/myosuite/issues/566)) | mjlab PPO action-std cap |
| [#587](https://github.com/MyoHub/myosuite/pull/587) | this teacher player + [tutorial 5.7](../../tutorials/5.7_ChaseTag_Steered_Teacher.ipynb) |

`mm-10m-2` needs `MYOSUITE_MUSCLEMIMIC_MODELS_VERSION=1.0.5`.

## Install

```bash
pip install -e ".[musclemimic]"
pip install -e training/chasetag_teacher
```

## Collect

```bash
MYOSUITE_MUSCLEMIMIC_MODELS_VERSION=1.0.5 \
  python -m chasetag_teacher.collect --episodes 4 --steps 400 --out pairs.npz
```

Use `pairs.npz` as BC data for a 546 → 354 student, or as an expert
dataset in #578. Do not change the scored FBVs observation.
