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
