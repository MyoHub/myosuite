# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Athletic task×model catalog and registration matrix.

``make_athletic(task)`` builds ``myofullbody`` with rigid tendons. Pass
``model="myofullbody_simple"`` for the reduced skeleton and
``tendons="elastic"`` for the series-elastic plugin. Tendon compliance is
not part of the Gymnasium id.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from myosuite.envs.myo.athletic.bodies import FULLBODY_SIMPLE_BODY_MAP, SPRINTER_BODY_MAP
from myosuite.envs.myo.athletic.factory import (
    make_athletic_env,
    mjlab_env_cfg_from_bundle,
    task_config_from_bundle,
)
from myosuite.envs.myo.athletic.specs import (
    DEFAULT_MODEL,
    DEFAULT_TENDONS,
    AthleticBundle,
    PlantSpec,
    TaskSpec,
    TendonKind,
    TermRef,
)
from myosuite.envs.myo.tasks.basic.full_body.athletic_contract import AthleticSpec

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plants
# ---------------------------------------------------------------------------


def _load_myosprinter():
    import myo_sim

    return myo_sim.load_spec("myosprinter")


def _load_myofullbody_simple():
    import myo_sim

    return myo_sim.load_spec("myofullbody_simple")


def _load_myofullbody():
    import myo_sim

    return myo_sim.load_spec("myofullbody")


PLANTS: dict[str, PlantSpec] = {
    "myosprinter": PlantSpec(
        key="myosprinter",
        entity_name="sprinter",
        model_recipe="myosprinter",  # mjlab-only plant; CPU recipe optional
        heading_dir=(1.0, 0.0),  # Bolt Y-up, forward +X
        up_axis="y",
        fall_height=0.55,  # match make_sprint_env_cfg FALL_HEIGHT (bit-match freeze)
        body_map=SPRINTER_BODY_MAP,
        action_kind="bolt_muscle",
        load_spec=_load_myosprinter,
        use_golden_sprint_cfg=True,
        env_id_prefix="",
        aliases=("myoSprint-v0", "Mjlab-Sprint-Flat-BoltMjWarp"),
    ),
    "myofullbody_simple": PlantSpec(
        key="myofullbody_simple",
        entity_name="fullbody_simple",
        model_recipe="myofullbody_simple",
        heading_dir=(-1.0, 0.0),  # Z-up composite faces −X
        up_axis="z",
        fall_height=0.6,
        body_map=FULLBODY_SIMPLE_BODY_MAP,
        action_kind="bolt_muscle",
        load_spec=_load_myofullbody_simple,
        use_golden_sprint_cfg=False,
        env_id_prefix="FullbodySimple",
        aliases=("myoFullbodySimpleSprint-v0",),
        capabilities=frozenset(
            {
                "lanes",
                "hop",
                "hurdles",
                "blocks",
                "curve",
                "vertical",
                "carioca",
                "locomotion",
                "imitate",
            }
        ),
    ),
    # Default skeleton. Rigid tendons are native mjMUSCLE; elastic selects
    # the plugin (AthleticBundle.action_kind) and is not a second plant key.
    "myofullbody": PlantSpec(
        key="myofullbody",
        entity_name="fullbody",
        model_recipe="myofullbody",
        heading_dir=(-1.0, 0.0),
        up_axis="z",
        fall_height=0.6,
        body_map=FULLBODY_SIMPLE_BODY_MAP,
        action_kind="native_muscle",
        load_spec=_load_myofullbody,
        use_golden_sprint_cfg=False,
        env_id_prefix="Fullbody",
        aliases=("myoFullbodySprint-v0",),
        capabilities=frozenset(
            {
                "lanes",
                "hop",
                "hurdles",
                "blocks",
                "curve",
                "vertical",
                "carioca",
                "locomotion",
                "imitate",
            }
        ),
    ),
}

# ---------------------------------------------------------------------------
# Tasks (msk_envs DerivedEnv sweep)
# ---------------------------------------------------------------------------

_LANE = AthleticSpec()


def _lane_rewards(spec: AthleticSpec = _LANE) -> tuple[TermRef, ...]:
    return (
        TermRef("rew_vel", weight=spec.lambda_vel),
        TermRef("rew_mid_lane", weight=spec.lambda_mid_lane),
        TermRef("rew_limit", weight=spec.lambda_limit),
        TermRef("rew_spring", weight=spec.lambda_spring),
        TermRef("rew_damper", weight=spec.lambda_damper),
        TermRef("rew_muscle_passive", weight=spec.lambda_muscle_passive),
    )


TASKS: dict[str, TaskSpec] = {
    "sprint": TaskSpec(
        key="sprint",
        kind="sprint",
        athletic=_LANE,
        heading_mode="forward",
        reward_terms=_lane_rewards(),
        scene="lanes",
        pose_key="starting_pose_run",
        requires=frozenset({"lanes"}),
    ),
    "backpedal": TaskSpec(
        key="backpedal",
        kind="backpedal",
        athletic=_LANE,
        heading_mode="backward",
        facing_angle_deg=15.0,
        reward_terms=_lane_rewards(),
        scene="lanes",
        pose_key="starting_pose_backpedal",
        requires=frozenset({"lanes"}),
    ),
    "side_shuffle": TaskSpec(
        key="side_shuffle",
        kind="side_shuffle",
        athletic=_LANE,
        heading_mode="lateral",
        reward_terms=_lane_rewards(),
        terminations=("time_out", "fallen", "not_facing", "toes_out", "feet_cross"),
        scene="lanes",
        pose_key="starting_pose_side",
        requires=frozenset({"lanes"}),
        extra={"no_feet_cross": True},
    ),
    "hop": TaskSpec(
        key="hop",
        kind="hop",
        athletic=_LANE,
        heading_mode="forward",
        reward_terms=_lane_rewards(),
        terminations=(
            "time_out",
            "fallen",
            "not_facing",
            "toes_out",
            "hop_left_toe",
            "hop_knee_straight",
        ),
        scene="lanes",
        pose_key="starting_pose_run",
        requires=frozenset({"lanes", "hop"}),
        extra={"stance_foot": "right"},
    ),
    "hurdles": TaskSpec(
        key="hurdles",
        kind="hurdles",
        athletic=_LANE,
        heading_mode="forward",
        reward_terms=_lane_rewards(),
        obs_keys=(
            "joint_pos",
            "joint_vel",
            "muscle_act",
            "root_planar_vel",
            "nearest_hurdle",
        ),
        scene="hurdles",
        pose_key="starting_pose_run",
        episode_length_s=12.0,
        requires=frozenset({"lanes", "hurdles"}),
    ),
    "sprint_blocks": TaskSpec(
        key="sprint_blocks",
        kind="sprint_blocks",
        athletic=_LANE,
        heading_mode="forward",
        reward_terms=_lane_rewards(),
        scene="blocks",
        pose_key="starting_pose_blockstart",
        requires=frozenset({"lanes", "blocks"}),
        extra={"apply_start_noise": False},
    ),
    "carioca": TaskSpec(
        key="carioca",
        kind="carioca",
        athletic=_LANE,
        heading_mode="lateral",
        reward_terms=_lane_rewards(),
        terminations=("time_out", "fallen", "not_facing", "toes_out", "wrong_cross"),
        scene="lanes",
        pose_key="starting_pose_side",
        requires=frozenset({"lanes", "carioca"}),
        extra={"fsm_phases": 4},
    ),
    "curve": TaskSpec(
        key="curve",
        kind="curve",
        athletic=_LANE,
        heading_mode="forward",
        reward_terms=(
            TermRef("rew_tangent_vel", weight=_LANE.lambda_vel),
            TermRef("rew_mid_lane", weight=_LANE.lambda_mid_lane),
            TermRef("rew_limit", weight=_LANE.lambda_limit),
        ),
        scene="curve",
        pose_key="starting_pose_run",
        requires=frozenset({"curve"}),
        extra={"lane_index": 0},
    ),
    "vertical": TaskSpec(
        key="vertical",
        kind="vertical",
        athletic=_LANE,
        heading_mode="forward",
        reward_terms=(
            TermRef("rew_vertical", weight=1.0),
            TermRef("rew_limit", weight=_LANE.lambda_limit),
        ),
        terminations=("time_out", "fallen"),
        obs_keys=("joint_pos", "joint_vel", "muscle_act", "root_planar_vel", "time"),
        scene="vertical",
        pose_key="starting_pose_run",
        requires=frozenset({"vertical"}),
    ),
    "locomotion": TaskSpec(
        key="locomotion",
        kind="locomotion",
        athletic=_LANE,
        heading_mode="forward",
        reward_terms=(
            TermRef("rew_track_vel", weight=1.0),
            TermRef("rew_limit", weight=_LANE.lambda_limit),
        ),
        terminations=("time_out", "fallen"),
        obs_keys=(
            "joint_pos",
            "joint_vel",
            "muscle_act",
            "command_vel",
        ),
        scene=None,
        pose_key="starting_pose_run",
        requires=frozenset({"locomotion"}),
        extra={
            "speed_min": 0.0,
            "speed_max": 4.0,
            "heading_range_deg": 180.0,
            "stand_prob": 0.1,
            "track_sigma": 0.5,
        },
    ),
    "imitate": TaskSpec(
        key="imitate",
        kind="imitate",
        athletic=_LANE,
        heading_mode="forward",
        reward_terms=(
            TermRef("rew_imitate_q", weight=1.0),
            TermRef("rew_imitate_u", weight=0.1),
            TermRef("rew_limit", weight=_LANE.lambda_limit),
        ),
        terminations=("time_out", "fallen"),
        obs_keys=("joint_pos", "joint_vel", "muscle_act", "ref_q", "ref_u", "time"),
        scene=None,
        pose_key="starting_pose_run",
        requires=frozenset({"imitate"}),
        extra={"clip_path": None},
    ),
    "uphill_sprint": TaskSpec(
        key="uphill_sprint",
        kind="uphill_sprint",
        athletic=_LANE,
        heading_mode="forward",
        reward_terms=_lane_rewards(),
        scene="uphill",
        pose_key="starting_pose_run",
        requires=frozenset({"lanes"}),
        extra={"ground_tilt_deg": 5.0},
    ),
}


def iter_bundles(
    tasks: Iterable[str] | None = None,
    plants: Iterable[str] | None = None,
) -> list[AthleticBundle]:
    task_keys = list(tasks) if tasks is not None else list(TASKS)
    plant_keys = list(plants) if plants is not None else list(PLANTS)
    out: list[AthleticBundle] = []
    for pk in plant_keys:
        plant = PLANTS[pk]
        for tk in task_keys:
            task = TASKS[tk]
            if not task.requires.issubset(plant.capabilities):
                continue
            out.append(AthleticBundle(task=task, plant=plant))
    return out


def _resolve_model(model: str | None, plant: str | None) -> str:
    """``model`` is the skeleton. ``plant`` is the previous name of that argument."""
    if model is not None and plant is not None and model != plant:
        raise ValueError("pass model or plant, not both")
    if model is not None:
        return model
    if plant is not None:
        return plant
    return DEFAULT_MODEL


def athletic_bundle(
    task: str,
    model: str | None = None,
    tendons: TendonKind = DEFAULT_TENDONS,
    *,
    plant: str | None = None,
) -> AthleticBundle:
    """Resolve a task, skeleton, and series element without building an env.

    ``model`` defaults to ``myofullbody``. ``tendons`` defaults to ``rigid``.
    """
    chosen = _resolve_model(model, plant)
    if task not in TASKS:
        raise ValueError(f"unknown athletic task {task!r}; expected one of {tuple(TASKS)}")
    if chosen not in PLANTS:
        raise ValueError(f"unknown athletic model {chosen!r}; expected one of {tuple(PLANTS)}")
    if tendons not in ("rigid", "elastic"):
        raise ValueError(f"unknown tendons {tendons!r}; expected 'rigid' or 'elastic'")
    return AthleticBundle(task=TASKS[task], plant=PLANTS[chosen], tendons=tendons)


def make_athletic(
    task: str,
    model: str | None = None,
    backend: str = "mjlab",
    *,
    tendons: TendonKind = DEFAULT_TENDONS,
    plant: str | None = None,
    **overrides: Any,
) -> Any:
    """Create an athletic env.

    ``make_athletic("sprint")`` is ``myofullbody`` with rigid tendons.
    ``make_athletic("sprint", "myofullbody_simple", tendons="elastic")``
    selects the reduced skeleton and the series-elastic plugin. ``plant``
    is the previous name of ``model``. Registered ids stay one per task and
    skeleton (``myoFullbodySprint-v0``, ``myoFullbodySimpleSprint-v0``);
    ``tendons`` is a constructor argument, not part of the id.
    """
    bundle = athletic_bundle(task, model, tendons, plant=plant)
    return make_athletic_env(bundle, backend=backend, **overrides)


def register_athletic_matrix(
    *,
    tasks: Iterable[str] | None = None,
    plants: Iterable[str] | None = None,
    backends: set[str] | None = None,
) -> list[str]:
    """Register CPU ModularTaskEnv and/or mjlab tasks for the athletic matrix.

    Returns list of registered env ids. Registration is always rigid tendons.
    Pass ``tendons`` to :func:`make_athletic` or ``gym.make`` to request the
    series element. Bolt ``myoSprint-v0`` keeps its golden
    ``make_sprint_env_cfg`` path via ``PlantSpec.use_golden_sprint_cfg``.
    """
    backends = backends or {"cpu", "mjlab"}
    registered: list[str] = []
    for bundle in iter_bundles(tasks=tasks, plants=plants):
        env_id = bundle.env_id()
        if "cpu" in backends and bundle.plant.model_recipe in _CPU_RECIPES:
            try:
                from myosuite.core.registry import register_env

                cfg = task_config_from_bundle(bundle)
                register_env(
                    env_id=env_id,
                    entry_point=(
                        "myosuite.envs.myo.athletic.factory:AthleticModularTaskEnv"
                    ),
                    max_episode_steps=cfg.max_episode_steps,
                    config=cfg.to_env_config(env_id=env_id),
                    kwargs={"task_config": cfg},
                )
                registered.append(env_id)
            except Exception:
                _LOG.warning("athletic CPU register failed for %s", env_id, exc_info=True)
        if "mjlab" in backends:
            try:
                from mjlab.tasks.registry import register_mjlab_task

                from myosuite.envs.myo.athletic.factory import sprint_ppo_runner_cfg

                env_cfg = mjlab_env_cfg_from_bundle(bundle)
                register_mjlab_task(
                    task_id=env_id,
                    env_cfg=env_cfg,
                    play_env_cfg=env_cfg,
                    rl_cfg=sprint_ppo_runner_cfg(bundle),
                    runner_cls=None,
                )
                registered.append(env_id)
                for alias in bundle.plant.aliases:
                    if alias == env_id:
                        continue
                    if bundle.task.key != "sprint":
                        continue
                    try:
                        register_mjlab_task(
                            task_id=alias,
                            env_cfg=env_cfg,
                            play_env_cfg=env_cfg,
                            rl_cfg=sprint_ppo_runner_cfg(bundle),
                            runner_cls=None,
                        )
                        registered.append(alias)
                    except ValueError:
                        pass
            except ValueError:
                pass  # already registered
            except Exception:
                _LOG.warning("athletic mjlab register failed for %s", env_id, exc_info=True)
    return registered


# CPU ModularTaskEnv only for plants that have a model recipe registered.
_CPU_RECIPES = frozenset({"myofullbody_simple", "myofullbody"})
