# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Model-generation detection, anatomical synergy maps, and ``(task, model, method)`` recipes.

Separates **task semantics** (scene, goal, observations, ground-truth ``solved`` /
``terminated`` conditions) from **model- and algorithm-dependent learning defaults**
(reward weights, action parameterization, scapulothoracic locking, RSI probability,
and PPO runner hyperparameters) across ``v0`` (``myo_sim 0.2.5``), ``v1``
(``myo_sim >= 0.3.0`` compliant & rigid), and custom musculoskeletal models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import mujoco
import numpy as np

from myosuite.physics.compliant_plant import detect_model_generation

RewardMode = Literal["default", "task_only"]
ActionStructure = Literal["buchanan", "sigmoid", "excitation", "synergy_residual"]

_SYNERGY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("hip_flex", ("iliacus", "psoas", "recfem", "sart", "tfl", "grac")),
    ("hip_ext", ("glmax", "bflh", "semimem", "semiten")),
    ("hip_abd", ("glmed", "glmin", "piri", "gem")),
    ("hip_add", ("addbrev", "addlong", "addmag", "pect")),
    ("knee_ext", ("vasmed", "vaslat", "vasint")),
    ("knee_flex", ("bfsh",)),
    ("ankle_pf", ("soleus", "gasmed", "gaslat", "tibpost", "fdl", "fhl", "perbrev", "perlong")),
    ("ankle_df", ("tibant", "edl", "ehl")),
    ("foot_intr", ("fdb", "abdh")),
    ("core_ext", ("ercspn", "multif", "quad_lumb", "iap_lumbar", "iliocost", "longissi")),
    ("core_flx", ("rect_abd", "intobl", "extobl")),
    ("neck_jaw", ("stern_mast", "scalen", "long_cap", "long_col", "splen", "semisp", "trap_cl", "levator", "masseter", "temporalis", "pteryg", "digastric", "hyoid")),
    ("scapula", ("trap", "serr", "rhom", "levs", "pecmin")),
    ("sh_ant", ("delt1", "pecm", "corb")),
    ("sh_lat", ("delt2", "supsp")),
    ("sh_post", ("delt3", "lat", "tmaj")),
    ("rot_cuff", ("infsp", "subsc", "tmin")),
    ("elbow_flex", ("biclong", "bicshort", "bra", "brd")),
    ("elbow_ext", ("trilong", "trilat", "trimed", "anc")),
    ("forearm_wr", ("sup", "pt_", "pq_", "ecrl", "ecrb", "ecu", "fcr", "fcu", "pl_")),
    ("thumb", ("op_", "apb", "adp", "fpb", "fpl", "epl", "epb", "apl")),
    ("fingers", ("fds", "fdp", "lum", "edc", "eip", "edm", "di", "pi", "fdm", "adm", "odm")),
)


@dataclass(frozen=True)
class AnatomicalSynergyMap:
    """Bilateral functional muscle-group projection ``(n_groups, nu)`` for any ``MjModel``.

    Attributes:
        group_names: Ordered synergy group names, shape ``(n_groups,)``.
        expand_matrix: Normalized row-averaged projection ``(n_groups, nu)`` mapping
            group drives to per-actuator logits.
        tonic_excitation: Baseline low-tonus excitation ``u0`` in ``(0, 1)``, shape ``(nu,)``.
    """

    group_names: tuple[str, ...]
    expand_matrix: np.ndarray
    tonic_excitation: np.ndarray


def _classify_actuator(name: str) -> str:
    low = name.lower()
    side = ""
    if low.endswith(("_r", "_right")):
        side = "_r"
    elif low.endswith(("_l", "_left")):
        side = "_l"
    for grp, prefixes in _SYNERGY_RULES:
        if any(low.startswith(p) or f"_{p}" in low for p in prefixes):
            return f"{grp}{side}"
    return f"act_{name}"


def build_anatomical_synergy_map(
    model: mujoco.MjModel,
    *,
    default_tonus: float = 0.16,
    distal_tonus: float = 0.06,
) -> AnatomicalSynergyMap:
    """Build a model-agnostic bilateral synergy map directly from *model*'s actuator names.

    Works identically across ``v0`` (``80M`` ``myolegs``, ``416M`` ``myofullbody``),
    ``v1`` (``90M`` ``myolegs``, ``172M/196M`` ``myofullbody_simple``, ``558M``
    ``myofullbody``), and custom models (falling back to 1-to-1 groups for unknown names).
    """
    nu = int(model.nu)
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"a{i}"
        for i in range(nu)
    ]
    grp_list: list[str] = []
    grp_to_idx: dict[str, int] = {}
    act_grp: list[int] = []
    tonus = np.full(nu, float(default_tonus), dtype=np.float32)

    for i, nm in enumerate(names):
        gn = _classify_actuator(nm)
        if gn not in grp_to_idx:
            grp_to_idx[gn] = len(grp_list)
            grp_list.append(gn)
        act_grp.append(grp_to_idx[gn])
        if gn.startswith(("thumb", "fingers", "neck_jaw")):
            tonus[i] = float(distal_tonus)

    n_grp = len(grp_list)
    mat = np.zeros((n_grp, nu), dtype=np.float32)
    for i, gi in enumerate(act_grp):
        mat[gi, i] = 1.0
    return AnatomicalSynergyMap(
        group_names=tuple(grp_list),
        expand_matrix=mat,
        tonic_excitation=tonus,
    )


@dataclass(frozen=True)
class TrainingRecipe:
    """Recommended ``(task_family, model_generation, method)`` configuration.

    Attributes:
        task_family: Canonical task family (``'walk'``, ``'waypoint'``, ``'mimic'``, ``'pose'``).
        model_generation: ``'v0'``, ``'v1_compliant'``, or ``'v1_rigid'``.
        method: Learning method key (``'ppo'``, ``'mimic_ppo'``, ``'task_only'``).
        reward_mode: ``'default'`` (shaped) or ``'task_only'`` (pure success/fail).
        reward_weights: Default dense reward term weights for this model generation.
        action_structure: Recommended action parameterization.
        lock_scapula_dofs: Whether to lock floating scapulothoracic DoFs on ``v1``
            whole-body locomotion/mimic when shoulder girdle mechanics are not studied.
        rsi_prob: Recommended Reference State Initialization fraction (``0.0`` when unused).
        runner_kwargs: Recommended PPO / RL runner hyperparameters.
    """

    task_family: str
    model_generation: str
    method: str = "ppo"
    reward_mode: RewardMode = "default"
    reward_weights: dict[str, float] = field(default_factory=dict)
    action_structure: ActionStructure = "buchanan"
    lock_scapula_dofs: bool = False
    rsi_prob: float = 0.0
    runner_kwargs: dict[str, Any] = field(default_factory=dict)


def resolve_training_recipe(
    task_family: str,
    model_or_generation: mujoco.MjModel | str = "v0",
    *,
    method: str = "ppo",
) -> TrainingRecipe:
    """Resolve the default ``(task, model, method)`` recipe.

    - ``v0`` (``myo_sim 0.2.5``) preserves the exact legacy Hugging Face baseline
      reward weights and direct per-muscle excitation contract.
    - ``v1`` (``myo_sim >= 0.3.0``, ``36 N/cm^2`` Handsfield specific tension +
      series-elastic tendons) reduces effort/fall local-minimum penalties and
      enables synergy-residual exploration + neutral scapular locking for whole-body
      locomotion from scratch.
    - ``method='task_only'`` sets ``reward_mode='task_only'`` (zero shaping).
    """
    gen = (
        detect_model_generation(model_or_generation)
        if isinstance(model_or_generation, mujoco.MjModel)
        else str(model_or_generation)
    )
    if method == "task_only":
        return TrainingRecipe(
            task_family=task_family,
            model_generation=gen,
            method="task_only",
            reward_mode="task_only",
            reward_weights={},
            action_structure="buchanan",
            lock_scapula_dofs=gen.startswith("v1"),
            rsi_prob=0.0,
            runner_kwargs={},
        )

    is_v1 = gen.startswith("v1")
    if task_family == "walk":
        if not is_v1:
            return TrainingRecipe(
                task_family="walk",
                model_generation=gen,
                method=method,
                reward_weights={
                    "vel_reward": 5.0,
                    "done": -100.0,
                    "cyclic_hip": -10.0,
                    "ref_rot": 10.0,
                    "joint_angle_rew": 5.0,
                },
                action_structure="buchanan",
                lock_scapula_dofs=False,
                rsi_prob=0.0,
                runner_kwargs={"learning_rate": 3e-4, "n_steps": 2048, "ent_coef": 0.001, "gamma": 0.99},
            )
        return TrainingRecipe(
            task_family="walk",
            model_generation=gen,
            method=method,
            reward_weights={
                "vel_reward": 8.0,
                "done": -15.0,
                "cyclic_hip": -6.0,
                "ref_rot": 8.0,
                "joint_angle_rew": 4.0,
                "act_mag": -0.5,
            },
            action_structure="synergy_residual",
            lock_scapula_dofs=True,
            rsi_prob=0.5,
            runner_kwargs={"learning_rate": 2.5e-4, "n_steps": 2048, "ent_coef": 0.0005, "gamma": 0.985},
        )

    if task_family == "waypoint":
        return TrainingRecipe(
            task_family="waypoint",
            model_generation=gen,
            method=method,
            reward_weights={"progress": 1.0, "arrival": 5.0, "effort": 0.0 if is_v1 else 0.01},
            action_structure="synergy_residual" if is_v1 else "buchanan",
            lock_scapula_dofs=is_v1,
            rsi_prob=0.0,
            runner_kwargs={"learning_rate": 2.5e-4, "n_steps": 2048, "ent_coef": 0.0005, "gamma": 0.99},
        )

    if task_family == "mimic":
        return TrainingRecipe(
            task_family="mimic",
            model_generation=gen,
            method=method,
            reward_weights={"scale": 0.1},
            action_structure="synergy_residual" if is_v1 else "excitation",
            lock_scapula_dofs=is_v1,
            rsi_prob=0.7,
            runner_kwargs={"learning_rate": 2.5e-4, "n_steps": 32, "ent_coef": 0.0005, "gamma": 0.985},
        )

    return TrainingRecipe(
        task_family=task_family,
        model_generation=gen,
        method=method,
        reward_weights={},
        action_structure="buchanan",
        lock_scapula_dofs=False,
        rsi_prob=0.0,
        runner_kwargs={"learning_rate": 3e-4},
    )
