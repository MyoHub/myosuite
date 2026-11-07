# myoSuperMimic (experimental)

SuperMimic for `myoMimicFullbody-v0`: a learned world model for the mechanics, with muscle
activation dynamics, forward kinematics, joint couplings and the task's observation, reward and
termination computed analytically, and an actor trained by backpropagating through imagined
rollouts. Code in `myosuite/learning/supermimic/`. One clip and the free-root full body only for
now; needs mjlab and a CUDA GPU (a 512-env run fits in about 8 GB).

## Clip

The runs use the KIT walking clip retargeted by MuscleMimic:

```python
from huggingface_hub import hf_hub_download

clip = hf_hub_download(
    "amathislab/musclemimic-retargeted",
    "MyoFullBody/gmr/KIT/167/walking_medium06_poses.npz",
    repo_type="dataset",
)
```

## Train

```bash
python scripts/supermimic/train.py --clip $CLIP --out runs/sm \
    --num-envs 512 --horizon 64 --imag-batch 512 --wm-batches 4 --seed 0
```

- The world model starts from scratch and is trained online. `--init-world-model` warm-starts it
  from an offline fit (`collect.py` records rollouts of an RSL-RL checkpoint with action noise;
  `train_world_model.py` fits and checks the model on them).
- `--ema-decay` (default 0.999) keeps an averaged actor. `--act-grad-clip 3` clips each sample's
  action gradient to 3 times the batch median (off by default). `--max-sim-steps` and
  `--max-hours` bound the run; `--resume` continues one.
- A run directory holds `ckpt_best.pt` (best training evaluation), `ckpt_final.pt` (last actor),
  `ckpt_ema.pt` (averaged actor), `log.jsonl` and `cfg.json`.

## InstantTrack changes (`--instanttrack`)

[InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack) (MIT) cuts SuperMimic's time to target on the
Unitree G1 by changes that are each a flag. Those that carry over to a muscle-driven body are in
`myosuite/learning/supermimic/instanttrack.py`; each has its own flag (off by default), and
`--instanttrack` sets InstantTrack's values:

| change | flag | preset |
|---|---|---|
| failure-focused resets: a share of resets starts one 1 s bin before recent termination points | `--fail-reset`, `--fail-bin-s` | 0.3 |
| foot-contact / anti-slip reward in imagination only (per-clip contact floor); the env's reward stays the metric | `--contact-reward` | 0.25 |
| Muon on the hidden weight matrices, Adam elsewhere | `--pi-opt muon --wm-opt muon` | on |
| learning rate, world-model batches per update, world-model size | `--lr`, `--wm-batches`, `--wm.hidden` | 4e-4, 5, 1024x3 |
| averaged actor | `--ema-decay` | 0.995 |
| imagined batch with a decayed policy learning-rate boost | `--imag-batch`, `--pi-lr-mult`, `--pi-lr-decay-at`, `--pi-lr-decay-len` | 2048, 4x, 1400, 300 |
| fused Adam, static-shape compilation | `--fused-adam`, `--static-compile` | on |

The preset keeps the horizon (64 beat 32 here). Not carried over: the reference-pose residual policy (no
joint-position targets for muscles), mirror augmentation of world-model batches (needs a muscle mirror
map), evaluation-guided resets, and the CUDA-graph / side-stream work.

```bash
python scripts/supermimic/train.py --clip $CLIP --out runs/sm_it --num-envs 512 --horizon 64 --instanttrack --seed 0
```

## Evaluate

```bash
# best, final and averaged actors of a run, on starts the training evaluation didn't use
python scripts/supermimic/eval_policies.py --clip $CLIP --run-dir runs/sm --out eval.json
# also works for PPO checkpoints in the RSL-RL layout
python scripts/supermimic/eval_policies.py --clip $CLIP path/to/model_80000.pt
# world-model drift against the simulator under the run's policy, at 8, 32 and 64 steps
python scripts/supermimic/wm_drift.py --clip $CLIP --checkpoint runs/sm/ckpt_final.pt
# side-by-side video of policies and the reference clip
python scripts/supermimic/render_compare.py --clip $CLIP --out compare.mp4 ...
```

Report the final or averaged actor: `ckpt_best.pt` is picked on the training evaluation, so its
training score is optimistic.
