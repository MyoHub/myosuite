# MuscleMimic training tools

Training-side tools for MyoSuite's [MuscleMimic](https://github.com/amathislab/musclemimic) integration: collecting
muscle activations and fitting SAR synergies, collecting behaviour-cloning data for the tutorial 5.4 directional
walker, and the ChaseTag steered teacher of tutorial 5.7.

This is a self-contained training project (`training/musclemimic/`, import name `musclemimic_training`), not part of
the installable `myosuite` package; see `docs/wiki/training-guidelines.md`. MyoSuite never imports it.

The MuscleMimic envs themselves stay in MyoSuite; they will be unified in the multi-clip tracking task `myoMimic-v0`.
These tools use MyoSuite's MuscleMimic integration as it is today.

## Attribution

The code in this project is MyoSuite code (header `Copyright (c) MyoSuite Authors`); it does not copy upstream code.
It builds on the following work, which should be cited when you use it:

| Work | What this project uses |
|---|---|
| MuscleMimic, Li et al., 2026, [arXiv:2603.25544](https://arxiv.org/abs/2603.25544), [amathislab/musclemimic](https://github.com/amathislab/musclemimic) | the `amathislab/mm-10m-2` tracking policy (rolled through MyoSuite's own loader), the MyoFullBody models (`musclemimic_models`) and the retargeted motions it was trained on |
| SAR, Berg, Caggiano and Kumar, RSS 2023, [arXiv:2307.03716](https://arxiv.org/abs/2307.03716) | the synergy pipeline `sar_fitting` fits (PCA whitening, FastICA, MinMax scaling) and the episode filter of `activation_collector` |

The MuscleMimic checkpoints, models and retargeted datasets keep their upstream licences (see their model and dataset
cards). The retargeted motions derive from [AMASS](https://amass.is.tue.mpg.de) and are for non-commercial research
under the AMASS licence; cite AMASS and MuscleMimic as the dataset cards ask. BibTeX:
`myosuite/integrations/musclemimic/CITATION.bib` and `myosuite.integrations.citations`.

## What lives where

| | |
|---|---|
| MyoSuite (`myosuite/integrations/musclemimic/`) | body models and model versions, checkpoint I/O, the local policy runner and `load_teacher_runner`, playback, preview, the eval CLI, the demo cache, terrain, the two-agent scene, `fbvs_player`, the SAR model (`sar_extraction`: `SynergyModel`, encode, decode, save, load) and its torch transform |
| MyoSuite envs and terms | the directional observation (`myosuite.envs.myo.tasks.mimic.directional_obs`), the role and arena headings (`myosuite.terms.multiplayer.heading`) |
| this project (`musclemimic_training/`) | `activation_collector`, `sar_fitting`, `bc_directional_collector`, `steering`, `steered_teacher`, `chasetag_pairs` |

## Install

```bash
pip install "myosuite[musclemimic]"   # or: pip install -e ".[musclemimic]" from a MyoSuite checkout
pip install "git+https://github.com/MyoHub/myosuite.git@dev#subdirectory=training/musclemimic"
# or, from a checkout: pip install -e training/musclemimic
# rolling amathislab/mm-10m-2 (Orbax checkpoint) also needs: pip install "training/musclemimic[teacher]"
```

Install MyoSuite and this project from the same branch. `amathislab/mm-10m-2` was trained on `musclemimic_models`
1.0.5: build its model with `model_versions.checkpoint_models_version("hf://amathislab/mm-10m-2")`, or set
`MYOSUITE_MUSCLEMIMIC_MODELS_VERSION=1.0.5` for envs that build their own config.

## SAR: collect activations and fit synergies (tutorial 5.5)

```python
from musclemimic_training.activation_collector import CollectionConfig, collect_activations
from musclemimic_training.sar_fitting import compute_vaf_curve, extract_synergies, select_n_synergies
from myosuite.integrations.musclemimic.sar_extraction import save_synergy_model

acts = collect_activations(policy, model, clips, CollectionConfig(n_episodes_per_clip=12))  # (T, n_muscles)
n_syn = select_n_synergies(compute_vaf_curve(acts, max_synergies=60), threshold=0.90)
save_synergy_model(extract_synergies(acts, n_syn), "synergy_model")
```

MyoSuite's SAR tasks (`register_mimic_mjlab_tasks_with_sar`, `register_directional_walk_sar`) read the saved folder
with `load_synergy_model`.

## Behaviour-cloning data for the directional walker (tutorial 5.4)

`bc_directional_collector.collect_bc_dataset(teacher_root, model, clips, BcCollectionConfig(...))` rolls `mm-10m-2`
through circular walking clips and records the 528-dim directional observation (`directional_obs`, heading set to
the clip's walking direction) with the teacher's muscle excitations.

## ChaseTag steered teacher (tutorial 5.7)

`steered_teacher.ChaseTagSteeredPair` drives both players of `myoChallengeChaseTagFBVs-v0` with `mm-10m-2` on 8
yaw-rotated copies of a walking clip: the chaser walks toward the opponent, the runner away, and headings turn off the
fence. The teacher runs on a shadow MuscleMimic body (1.0.5), not on the scored body (myo_sim's `myofullbody`).

```python
import os
os.environ["MYOSUITE_MUSCLEMIMIC_MODELS_VERSION"] = "1.0.5"

from myosuite import make_env
from musclemimic_training.steered_teacher import ChaseTagSteeredPair

env = make_env("myoChallengeChaseTagFBVs-v0")
pair = ChaseTagSteeredPair.load(env)
obs, info = env.reset(seed=0)
pair.reset(env.unwrapped.model, env.unwrapped.data, env.unwrapped._meta, info["task"])
acts = pair.actions(env.unwrapped.model, env.unwrapped.data, env.unwrapped._meta, info["task"], 0)
```

`chasetag_pairs` writes the teacher's (546-d observation, 354 excitations) pairs:

```bash
MYOSUITE_MUSCLEMIMIC_MODELS_VERSION=1.0.5 python -m musclemimic_training.chasetag_pairs --episodes 4 --steps 400 --out pairs.npz
```

For a ChaseTag player trained on the scored body, see `training/chasetag_supermimic`.

## Tests

```bash
cd training/musclemimic && pytest
```

The tests use mocks and synthetic arrays; they need no checkpoint or motion data. MyoSuite's CI does not run them.
