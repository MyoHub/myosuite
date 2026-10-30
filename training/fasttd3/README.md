# FastTD3 on MyoSuite

Off-policy trainer for MyoSuite CPU and mjlab twins. Uses the official
[FastTD3](https://github.com/younggyoseo/FastTD3) Actor, distributional Critic,
and `SimpleReplayBuffer` (Seo et al., MIT). Env wrappers and the MuscleMimic
wiring are MyoSuite-specific; the update loop is adapted from FastTD3 `train.py`.

This project lives under `training/fasttd3/` and is **not** part of the
installable `myosuite` package. See [PR #581](https://github.com/MyoHub/myosuite/pull/581).

## Attribution

- FastTD3: Younggyo Seo and collaborators, [younggyoseo/FastTD3](https://github.com/younggyoseo/FastTD3) (MIT, 2025).
- LeanRL / CleanRL lineage: see `NOTICE`.

## Install

From a MyoSuite checkout (separate venv if the stack conflicts with other methods):

```bash
pip install -e .[musclemimic]
pip install -e training/fasttd3
```

mjlab GPU training also needs the Warp extra:

```bash
pip install -e .[mjlab,musclemimic]
pip install -e "./training/fasttd3[mjlab]"
```

From git only:

```bash
pip install myosuite
pip install "git+https://github.com/MyoHub/myosuite.git@dev#subdirectory=training/fasttd3"
```

## Run

```bash
python -m myosuite_fasttd3 myoMimicFullbody-v0 --smoke --backend cpu
python -m myosuite_fasttd3 myoMimicFullbody-v0 --backend mjlab --num-envs 1024
# or:
train-fasttd3 myoMimicFullbody-v0 --smoke --backend cpu
```

`--backend cpu` is Gymnasium (Colab smoke, no Warp, no Hugging Face clip).
`--backend mjlab` is the Warp twin. `--clip walk.npz` switches MuscleMimic to
trajectory mode. Muscle activations map FastTD3 `tanh` `[-1, 1]` to `[0, 1]`.

Colab T4: `--backend mjlab --num-envs 64` and leave AMP off. `--amp` is bf16-only
(A100/H100). fp16 AMP on unnormalized muscle observations overflows.

## Tests

```bash
pip install -e ".[musclemimic]" -e "./training/fasttd3[dev]"
pytest training/fasttd3/tests -v
```
