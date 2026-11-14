# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Multi-clip full-body motion tracking (``myoMimic-v0``).

The task (what is to be done) is separate from the body model and from the learning
method:

* task: track the AMASS KIT clips retargeted onto the task model (MuscleMimic's KIT
  split: 972 training, 108 held-out clips; :func:`kit_root`) with a full body (free root, legs, torso, arms; 17 tracked bodies
  and 4 foot bodies). Default observations from a shared vocabulary
  (:mod:`myosuite.terms.mimic_track`), a proposed dense reward (MimicReward), a
  failure rule (1 m relative sites / 1 m root / 1 rad root orientation) and success
  metrics that do not depend on the control step (:mod:`.metrics`);
* model: a body adapter (:mod:`.models`: myo_sim's ``myofullbody``, the fixed-base
  ``myoelbow2dof``, or a submodel added with ``register_model``), chosen with
  ``EnvConfig.model``;
* method: presets (:mod:`.config`) select a method's MDP choices (``"supermimic"``,
  the default); training methods live outside the package, in ``training/<method>/``.
  A method may train on its own
  reward (``overrides={"reward_fn": fn}``, ``fn(state, ref, ctrl_dt)``, or a whole
  ``RewardCfg`` as ``overrides={"reward": ...}``); the proposed reward is still reported
  (``info["task_reward"]`` on the CPU, the twin command's ``task_reward`` and its
  ``Metrics/motion/task_reward`` log) and is what the metrics score.

Build it with ``make_env(MIMIC_ENV_ID)`` on the CPU or ``backend="mjlab"``; the control
step is ``EnvConfig.ctrl_dt`` (default 0.02 s).
"""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path

MIMIC_ENV_ID = "myoMimic-v0"
"""Id of the task (rename here only)."""
DATASET_ENV = "MYOSUITE_MIMIC_DATASET"
"""Environment variable naming a Hugging Face dataset of retargeted clips, laid out as
``<model>/<version>/KIT/...`` (optional)."""
KIT_CLIP_SETS = {"kit_train": "train"}
KIT_ROOT_ENV = "MYOSUITE_KIT_ROOT"
"""Environment variable naming a local folder of retargeted clips (the root of the
packaged lists)."""


class MissingMotionData(FileNotFoundError):
    """The task's motion clips are not on this machine (they need the AMASS licence)."""


def kit_split(split: str) -> Path:
    """The packaged default clip list: MuscleMimic's KIT training motions (972 clips),
    the clips its released checkpoints were trained on. Held-out or other lists
    belong to the training projects that use them (``training/<method>/``).

    Args:
        split: ``"train"``.

    Returns:
        The ``.txt`` list; its paths are relative to the clip root.
    """
    if split != "train":
        raise ValueError(f"only the 'train' list is packaged, not {split!r}")
    return Path(str(files(__name__) / "data" / f"kit_{split}.txt"))


def download_retargeted_clips(
    repo_id: str, model: str, version: str, local_dir: str | None = None
) -> Path:
    """Download the KIT clips retargeted onto *model* at *version* from a Hugging Face
    dataset. Downloaded files are reused.

    Args:
        repo_id: The dataset.
        model: Task model name (e.g. ``"myofullbody"``).
        version: Its version (the myo_sim version).
        local_dir: Download into this directory instead of the Hugging Face cache.

    Returns:
        The ``<model>/<version>`` folder (the root of the packaged lists).
    """
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    base = snapshot_download(
        repo_id, repo_type="dataset",
        allow_patterns=f"{model}/{version}/*", local_dir=local_dir,
    )  # fmt: skip
    return Path(base) / model / version


def kit_root(model: str = "myofullbody") -> Path:
    """Root of the KIT clips: ``$MYOSUITE_KIT_ROOT``, else the local retarget cache,
    else the dataset ``$MYOSUITE_MIMIC_DATASET``.

    The cache is ``~/.cache/myosuite/mimic/<model>-<version>`` of the installed
    myo_sim version (``scripts/mimic/retarget_amass.py`` fills it from an AMASS
    download). Without it, a sibling ``<model>-<other version>`` folder is used whose
    first listed clip carries the installed body's kinematic fingerprint
    (:func:`~.models.kinematic_fingerprint`): a myo_sim release that leaves the body's
    kinematics alone keeps its clips (:func:`~myosuite.biomechanics.retarget.find_cache_root`).

    Raises:
        MissingMotionData: If none of them holds the clips (naming the folders tried).
    """
    if os.environ.get(KIT_ROOT_ENV):
        return Path(os.environ[KIT_ROOT_ENV])
    from myosuite.biomechanics.retarget import find_cache_root  # noqa: PLC0415

    from .models import kinematic_fingerprint, model_version  # noqa: PLC0415

    version = model_version(model)
    rel = next(line for line in kit_split("train").read_text().splitlines()
               if line.strip() and not line.startswith("#"))  # fmt: skip
    cache, tried = find_cache_root(model, version, kinematic_fingerprint(model), rel)
    if cache is not None:
        return cache
    if os.environ.get(DATASET_ENV):
        return download_retargeted_clips(os.environ[DATASET_ENV], model, version)
    raise MissingMotionData(
        f"no KIT clips for {model} (myo_sim {version}, kinematic fingerprint "
        f"{kinematic_fingerprint(model)}) in {', '.join(map(str, tried))}: run "
        f"scripts/mimic/retarget_amass.py on the AMASS KIT motions (fills {tried[0]}), "
        f"or set {KIT_ROOT_ENV}"
    )


def resolve_clips(
    clips: str, root: str = "", model: str = "myofullbody"
) -> tuple[str, str]:
    """``(path, root)`` of a :class:`~.config.MotionCfg` clip set.

    The KIT names map to the packaged lists; their root defaults to
    :func:`kit_root`. Anything else is a clip, directory or list.
    """
    if clips in KIT_CLIP_SETS:
        root = root or str(kit_root(model))
        return str(kit_split(KIT_CLIP_SETS[clips])), root
    return clips, root
