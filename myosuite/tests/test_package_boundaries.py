# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Enforce internal subpackage ownership and the upward module import DAG."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.tier1

_REPO = Path(__file__).resolve().parents[2]
_PACKAGE = _REPO / "myosuite"

# Active runtime subpackages in myosuite/.
_APPROVED_RUNTIME_SUBPACKAGES = frozenset(
    {
        "core",
        "physics",
        "terms",
        "scenes",
        "motion",
        "policies",
        "biomechanics",
        "integrations",
        "envs",
        "viz",
        "benchmarks",
        "logger",
        "utils",
    }
)

# Newly created subpackages must never accumulate legacy import exemptions.
_ZERO_EXEMPTION_SUBPACKAGES = frozenset({"motion", "policies", "benchmarks"})

# Out-of-wheel repository directories that runtime modules must never import.
_FORBIDDEN_EXTERNAL_IMPORTS = frozenset(
    {
        "training",
        "tools",
        "scripts",
        "myosuite_ppo",
        "myosuite_dynsyn",
        "supermimic",
        "chasetag_teacher",
    }
)

# Allowed runtime imports between myosuite.<src> and myosuite.<dst> subpackages.
# "__root__" denotes myosuite/__init__.py and myosuite/version.py.
_ALLOWED_INTERNAL_IMPORTS: dict[str, frozenset[str]] = {
    # Legacy compatibility subpackages on `__root__` support lazy `import myosuite; myosuite.<pkg>.*`.
    "__root__": frozenset(
        {"__root__", "core", "envs", "logger", "scenes", "terms", "utils"}
    ),
    # Foundation & numerical helpers
    "core": frozenset({"__root__", "core", "physics"}),
    "physics": frozenset({"__root__", "physics"}),
    "scenes": frozenset({"__root__", "core", "scenes"}),
    "terms": frozenset({"__root__", "core", "physics", "terms"}),
    "motion": frozenset({"__root__", "core", "physics", "motion"}),
    "policies": frozenset(
        {"__root__", "core", "terms", "physics", "motion", "biomechanics", "policies"}
    ),
    # Solvers & lightweight source/ecosystem adapters
    "biomechanics": frozenset(
        {"__root__", "core", "physics", "motion", "biomechanics"}
    ),
    "integrations": frozenset(
        {
            "__root__",
            "core",
            "terms",
            "physics",
            "motion",
            "policies",
            "biomechanics",
            "integrations",
        }
    ),
    # Simulation environments, visualization & evaluation suites
    "envs": frozenset(
        {
            "__root__",
            "core",
            "scenes",
            "terms",
            "physics",
            "motion",
            "policies",
            "biomechanics",
            "integrations",
            "envs",
        }
    ),
    "viz": frozenset(
        {
            "__root__",
            "core",
            "scenes",
            "physics",
            "motion",
            "policies",
            "biomechanics",
            "envs",
            "viz",
        }
    ),
    "benchmarks": frozenset(
        {
            "__root__",
            "core",
            "terms",
            "physics",
            "motion",
            "policies",
            "biomechanics",
            "integrations",
            "envs",
            "viz",
            "benchmarks",
        }
    ),
    # Closed backward-compatibility facade namespaces
    "logger": frozenset({"__root__", "core", "physics", "motion", "viz", "logger"}),
    "utils": frozenset(
        {
            "__root__",
            "core",
            "terms",
            "physics",
            "motion",
            "policies",
            "biomechanics",
            "envs",
            "viz",
            "benchmarks",
            "utils",
        }
    ),
}

# Lazy backward-compatibility re-exports on core/ preserving former public import paths until v4.0.
_COMPATIBILITY_REEXPORT_IMPORTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("core/__init__.py", "integrations"),
        ("core/__init__.py", "motion"),
        ("core/body_scaling.py", "integrations"),
    }
)

# Pre-existing cross-layer runtime edges on dev (14 edges), burned down as migrations land.
# Do not add entries here; place new runtime logic in its canonical layer.
_LEGACY_CROSS_IMPORTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("biomechanics/retarget_instanthmr.py", "viz"),
        ("core/config.py", "envs"),
        ("core/model_builder.py", "envs"),
        ("core/model_recipes.py", "envs"),
        ("core/model_recipes.py", "integrations"),
        ("core/registry.py", "envs"),
        ("envs/gymnasium_env.py", "viz"),
        ("envs/heightfields.py", "viz"),
        ("envs/multi_agent_modular_env.py", "viz"),
        ("integrations/musclemimic/fbvs_player.py", "envs"),
        ("integrations/musclemimic/fullbody_native_playback.py", "envs"),
        ("integrations/musclemimic/fullbody_native_playback.py", "viz"),
        ("integrations/musclemimic/mjlab_onnx_policy.py", "envs"),
        ("viz/mjlab_play.py", "integrations"),
    }
)

_COMPATIBILITY_MODULE_SHIMS: tuple[tuple[str, str], ...] = (
    ("myosuite.scenes.library", "myosuite.core.builder.scene_library"),
    ("myosuite.scenes.scene_builder", "myosuite.core.builder.scene_builder"),
    ("myosuite.envs.myo.myoedits.model_editor", "myosuite.core.builder.model_editor"),
    ("myosuite.utils.spec_processing", "myosuite.core.builder.spec_processing"),
    ("myosuite.utils.mujoco_geom_utils", "myosuite.core.builder.mujoco_geom_utils"),
    ("myosuite.utils.mujoco_joint_utils", "myosuite.core.builder.mujoco_joint_utils"),
    ("myosuite.utils.asset_path_resolver", "myosuite.core.builder.asset_path_resolver"),
    ("myosuite.utils.colored_noise", "myosuite.physics.colored_noise"),
    ("myosuite.terms._array", "myosuite.core.terms._array"),
    ("myosuite.terms.base_action", "myosuite.core.terms.base_action"),
    ("myosuite.terms.base_event", "myosuite.core.terms.base_event"),
    ("myosuite.terms.base_obs", "myosuite.core.terms.base_obs"),
    ("myosuite.terms.base_reward", "myosuite.core.terms.base_reward"),
    ("myosuite.terms.base_termination", "myosuite.core.terms.base_termination"),
    ("myosuite.terms.mimic_obs", "myosuite.core.terms.mimic_obs"),
    ("myosuite.terms.mimic_reward", "myosuite.core.terms.mimic_reward"),
    ("myosuite.terms.mimic_track", "myosuite.core.terms.mimic_track"),
    (
        "myosuite.terms.opponent_relative_obs",
        "myosuite.core.terms.opponent_relative_obs",
    ),
    ("myosuite.terms.waypoint", "myosuite.core.terms.waypoint"),
    ("myosuite.terms.multiplayer.common", "myosuite.core.terms.multiplayer.common"),
    ("myosuite.terms.multiplayer.heading", "myosuite.core.terms.multiplayer.heading"),
    ("myosuite.terms.catalog", "myosuite.benchmarks.term_catalog"),
    ("myosuite.core.trajectory_io", "myosuite.motion.io"),
    ("myosuite.core.trajectory_yaw", "myosuite.motion.transforms"),
    ("myosuite.core.playback_contract", "myosuite.motion.playback"),
    ("myosuite.core.hf_dataset_cache", "myosuite.motion.hf_cache"),
    ("myosuite.core.retargeting", "myosuite.integrations.marker_retarget"),
    ("myosuite.logger.reference_motion", "myosuite.motion.reference"),
    ("myosuite.logger.reference_motion_jax", "myosuite.motion.reference_jax"),
    ("myosuite.logger.examine_logs", "myosuite.viz.examine_logs"),
    ("myosuite.logger.examine_reference", "myosuite.viz.examine_reference"),
    ("myosuite.logger.grouped_datasets", "myosuite.viz.grouped_datasets"),
    ("myosuite.logger.roboset_logger", "myosuite.viz.roboset_logger"),
    ("myosuite.biomechanics.opensim_motion", "myosuite.integrations.opensim_motion"),
    ("myosuite.biomechanics.opensim_grf", "myosuite.integrations.opensim_grf"),
    ("myosuite.biomechanics.smplh", "myosuite.integrations.smplh"),
    ("myosuite.biomechanics.retarget_smplh", "myosuite.integrations.retarget_smplh"),
    ("myosuite.biomechanics.mano", "myosuite.integrations.mano"),
    ("myosuite.biomechanics.gvhmr", "myosuite.integrations.gvhmr"),
    ("myosuite.biomechanics.chumpy_pickle", "myosuite.integrations.chumpy_pickle"),
    ("myosuite.utils.action_map", "myosuite.policies.action_map"),
    ("myosuite.utils.checkpoint_manifest", "myosuite.policies.checkpoint_manifest"),
    ("myosuite.utils.checkpoint_utils", "myosuite.policies.checkpoint_utils"),
    ("myosuite.utils.deployment_manifest", "myosuite.policies.deployment_manifest"),
    ("myosuite.utils.export_onnx", "myosuite.policies.export_onnx"),
    ("myosuite.utils.onnx_checkpoint", "myosuite.policies.onnx_checkpoint"),
    ("myosuite.utils.onnxruntime_utils", "myosuite.policies.onnxruntime_utils"),
    ("myosuite.utils.rslrl_policy", "myosuite.policies.rslrl_policy"),
    ("myosuite.integrations.musclemimic.actor_tree", "myosuite.policies.actor_tree"),
    (
        "myosuite.integrations.musclemimic.fullbody_checkpoint_io",
        "myosuite.policies.checkpoint_ref",
    ),
    (
        "myosuite.integrations.musclemimic.fullbody_eval_cli",
        "myosuite.benchmarks.musclemimic_fullbody_eval_cli",
    ),
    ("myosuite.utils.examine_env", "myosuite.benchmarks.examine_env"),
    ("myosuite.utils.examine_sim", "myosuite.benchmarks.examine_sim"),
    ("myosuite.utils.env_check", "myosuite.benchmarks.env_check"),
    ("myosuite.utils.paired_replay", "myosuite.benchmarks.paired_replay"),
    ("myosuite.utils.performance_card", "myosuite.benchmarks.performance_card"),
    ("myosuite.utils.push_metrics", "myosuite.benchmarks.push_metrics"),
    ("myosuite.utils.push_protocol", "myosuite.benchmarks.push_protocol"),
    ("myosuite.utils.push_recovery", "myosuite.benchmarks.push_recovery"),
    ("myosuite.utils.push_results", "myosuite.benchmarks.push_results"),
    ("myosuite.utils.push_scheduling", "myosuite.benchmarks.push_scheduling"),
    ("myosuite.utils.dict_utils", "myosuite.core.dict_utils"),
    ("myosuite.utils.import_utils", "myosuite.core.import_utils"),
    ("myosuite.utils.path_utils", "myosuite.core.path_utils"),
    ("myosuite.utils.prompt_utils", "myosuite.core.prompt_utils"),
    ("myosuite.utils.reach_workspace", "myosuite.core.reach_workspace"),
    ("myosuite.utils.step_timing", "myosuite.core.step_timing"),
    ("myosuite.utils.target_ranges", "myosuite.core.target_ranges"),
    ("myosuite.utils.tensor_utils", "myosuite.core.tensor_utils"),
    ("myosuite.utils.feature_cli", "myosuite.envs.feature_cli"),
    ("myosuite.utils.html", "myosuite.viz.html_embed"),
    ("myosuite.utils.path_plotting", "myosuite.viz.path_plotting"),
    ("myosuite.utils.policy_utils", "myosuite.viz.policy_playback"),
    ("myosuite.utils.video_io", "myosuite.viz.video_io"),
    ("myosuite.envs.waypoint", "myosuite.envs.tasks.fullbody.waypoint.env"),
    (
        "myosuite.terms.multiplayer.chase_tag_vs_obs",
        "myosuite.envs.tasks.games.chasetag.chase_tag_vs_obs",
    ),
    (
        "myosuite.terms.multiplayer.chase_tag_vs_reward",
        "myosuite.envs.tasks.games.chasetag.chase_tag_vs_reward",
    ),
    ("myosuite.envs.myo.assets._resolve", "myosuite.envs.assets._resolve"),
    (
        "myosuite.envs.myo.assets.leg.myoosl_control",
        "myosuite.envs.assets.leg.myoosl_control",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.clip_trajectory_source",
        "myosuite.envs.backends.mjlab.clip_trajectory_source",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.configs.asset_configs.muscle_mimic_body.full_body_constants",
        "myosuite.envs.backends.mjlab.configs.asset_configs.muscle_mimic_body.full_body_constants",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.configs.baoding_cfg",
        "myosuite.envs.tasks.hand.baoding.mjlab_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.configs.elbow_pose_cfg",
        "myosuite.envs.backends.mjlab.configs.elbow_pose_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.configs.finger_pose_cfg",
        "myosuite.envs.backends.mjlab.configs.finger_pose_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.configs.finger_reach_cfg",
        "myosuite.envs.backends.mjlab.configs.finger_reach_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.configs.hand_pose_cfg",
        "myosuite.envs.backends.mjlab.configs.hand_pose_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.configs.hand_reach_cfg",
        "myosuite.envs.backends.mjlab.configs.hand_reach_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.configs.modular_cfg",
        "myosuite.envs.backends.mjlab.configs.modular_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.configs.musclemimic_bimanual_cfg",
        "myosuite.envs.tasks.fullbody.bimanual.mjlab_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.configs.musclemimic_fullbody_cfg",
        "myosuite.envs.tasks.mimic.fullbody_mjlab_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.configs.table_tennis_cfg",
        "myosuite.envs.tasks.games.tabletennis.mjlab_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.configs.walk_cfg",
        "myosuite.envs.backends.mjlab.configs.walk_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.mimic_mjlab_env",
        "myosuite.envs.backends.mjlab.mimic_mjlab_env",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.mjlab_env_base",
        "myosuite.envs.backends.mjlab.mjlab_env_base",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.mjlab_task_builder",
        "myosuite.envs.backends.mjlab.mjlab_task_builder",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.register_mjlab_tabletennis",
        "myosuite.envs.tasks.games.tabletennis.register_mjlab",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.register_mjlab_tasks",
        "myosuite.envs.backends.mjlab.register_mjlab_tasks",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.rsl_rl_logger_episode_patch",
        "myosuite.envs.backends.mjlab.rsl_rl_logger_episode_patch",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.cpu_reference",
        "myosuite.envs.backends.mjlab.tasks.cpu_reference",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.distribution",
        "myosuite.envs.backends.mjlab.tasks.distribution",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.leg.directional_env_cfg",
        "myosuite.envs.tasks.leg.directional.mjlab_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.leg.directional_mdp",
        "myosuite.envs.tasks.leg.directional.directional_mdp",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.leg.rl_cfg",
        "myosuite.envs.tasks.leg.walk.rl_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.leg.stand_env_cfg",
        "myosuite.envs.tasks.leg.stand.mjlab_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.leg.terrain",
        "myosuite.envs.tasks.leg.walk.terrain",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.leg.walk_env_cfg",
        "myosuite.envs.tasks.leg.walk.mjlab_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.leg.walk_mdp",
        "myosuite.envs.tasks.leg.walk.walk_mdp",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.mdp.action_map",
        "myosuite.envs.backends.mjlab.tasks.mdp.action_map",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.mdp.actions",
        "myosuite.envs.backends.mjlab.tasks.mdp.actions",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.mdp.commands",
        "myosuite.envs.backends.mjlab.tasks.mdp.commands",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.mdp.events",
        "myosuite.envs.backends.mjlab.tasks.mdp.events",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.mdp.observations",
        "myosuite.envs.backends.mjlab.tasks.mdp.observations",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.mdp.terminations",
        "myosuite.envs.backends.mjlab.tasks.mdp.terminations",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.mimic_tracking.commands",
        "myosuite.envs.tasks.mimic.mjlab_commands",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.mimic_tracking.env_cfg",
        "myosuite.envs.tasks.mimic.mjlab_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.mimic_tracking.mdp",
        "myosuite.envs.tasks.mimic.mjlab_mdp",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.pose.config.elbow.env_cfgs",
        "myosuite.envs.tasks.arm.pose.mjlab_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.pose.config.elbow.rl_cfg",
        "myosuite.envs.tasks.arm.pose.rl_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.pose.config.finger.env_cfgs",
        "myosuite.envs.tasks.hand.pose.finger_mjlab_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.pose.config.finger.rl_cfg",
        "myosuite.envs.tasks.hand.pose.finger_rl_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.pose.config.hand.env_cfgs",
        "myosuite.envs.tasks.hand.pose.hand_mjlab_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.pose.config.hand.rl_cfg",
        "myosuite.envs.tasks.hand.pose.hand_rl_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.pose.config.torso.env_cfgs",
        "myosuite.envs.tasks.torso.pose.mjlab_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.pose.config.torso.rl_cfg",
        "myosuite.envs.tasks.torso.pose.rl_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.pose.mdp.commands",
        "myosuite.envs.backends.mjlab.tasks.pose.mdp.commands",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.pose.mdp.observations",
        "myosuite.envs.backends.mjlab.tasks.pose.mdp.observations",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.pose.mdp.rewards",
        "myosuite.envs.backends.mjlab.tasks.pose.mdp.rewards",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.pose.mdp.terminations",
        "myosuite.envs.backends.mjlab.tasks.pose.mdp.terminations",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.pose.pose_env_cfg",
        "myosuite.envs.backends.mjlab.tasks.pose.pose_env_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.reach.config.arm.env_cfgs",
        "myosuite.envs.tasks.arm.reach.mjlab_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.reach.config.arm.rl_cfg",
        "myosuite.envs.tasks.arm.reach.rl_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.reach.config.finger.env_cfgs",
        "myosuite.envs.tasks.hand.reach.finger_mjlab_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.reach.config.finger.rl_cfg",
        "myosuite.envs.tasks.hand.reach.finger_rl_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.reach.config.hand.env_cfgs",
        "myosuite.envs.tasks.hand.reach.hand_mjlab_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.reach.config.hand.rl_cfg",
        "myosuite.envs.tasks.hand.reach.hand_rl_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.reach.mdp.commands",
        "myosuite.envs.backends.mjlab.tasks.reach.mdp.commands",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.reach.mdp.observations",
        "myosuite.envs.backends.mjlab.tasks.reach.mdp.observations",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.reach.mdp.rewards",
        "myosuite.envs.backends.mjlab.tasks.reach.mdp.rewards",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.reach.mdp.terminations",
        "myosuite.envs.backends.mjlab.tasks.reach.mdp.terminations",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.reach.reach_env_cfg",
        "myosuite.envs.backends.mjlab.tasks.reach.reach_env_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.registration",
        "myosuite.envs.backends.mjlab.tasks.registration",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.rl",
        "myosuite.envs.backends.mjlab.tasks.rl",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.waypoint.env_cfg",
        "myosuite.envs.tasks.fullbody.waypoint.mjlab_cfg",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.waypoint.mdp",
        "myosuite.envs.tasks.fullbody.waypoint.mdp",
    ),
    (
        "myosuite.envs.myo.backends.mjx.brax_env",
        "myosuite.envs.backends._experimental_mjx.brax_env",
    ),
    (
        "myosuite.envs.myo.backends.mjx.fatigue_jax",
        "myosuite.envs.backends._experimental_mjx.fatigue_jax",
    ),
    (
        "myosuite.envs.myo.backends.mjx.mjx_env_base",
        "myosuite.envs.backends._experimental_mjx.mjx_env_base",
    ),
    (
        "myosuite.envs.myo.backends.mjx.mjx_env_config",
        "myosuite.envs.backends._experimental_mjx.mjx_env_config",
    ),
    (
        "myosuite.envs.myo.backends.mjx.mjx_modular_env",
        "myosuite.envs.backends._experimental_mjx.mjx_modular_env",
    ),
    (
        "myosuite.envs.myo.backends.mjx.mjx_spec_preprocess",
        "myosuite.envs.backends._experimental_mjx.mjx_spec_preprocess",
    ),
    (
        "myosuite.envs.myo.backends.mjx.musclemimic_base",
        "myosuite.envs.backends._experimental_mjx.musclemimic_base",
    ),
    (
        "myosuite.envs.myo.backends.mjx.musclemimic_env",
        "myosuite.envs.backends._experimental_mjx.musclemimic_env",
    ),
    (
        "myosuite.envs.myo.backends.mjx.musclemimic_fullbody_env",
        "myosuite.envs.backends._experimental_mjx.musclemimic_fullbody_env",
    ),
    (
        "myosuite.envs.myo.backends.mjx.pose_env",
        "myosuite.envs.backends._experimental_mjx.pose_env",
    ),
    (
        "myosuite.envs.myo.backends.mjx.reach_env",
        "myosuite.envs.backends._experimental_mjx.reach_env",
    ),
    (
        "myosuite.envs.myo.backends.mjx.train_jax_ppo",
        "myosuite.envs.backends._experimental_mjx.train_jax_ppo",
    ),
    (
        "myosuite.envs.myo.backends.mjx.utils",
        "myosuite.envs.backends._experimental_mjx.utils",
    ),
    (
        "myosuite.envs.myo.backends.mjx.visu_mjx_env",
        "myosuite.envs.backends._experimental_mjx.visu_mjx_env",
    ),
    (
        "myosuite.envs.myo.backends.mjx.walk_env",
        "myosuite.envs.backends._experimental_mjx.walk_env",
    ),
    (
        "myosuite.envs.myo.tasks.basic.arm.key_turn",
        "myosuite.envs.tasks.arm.key_turn.env",
    ),
    ("myosuite.envs.myo.tasks.basic.arm.pose", "myosuite.envs.tasks.arm.pose.env"),
    ("myosuite.envs.myo.tasks.basic.arm.reach", "myosuite.envs.tasks.arm.reach.env"),
    (
        "myosuite.envs.myo.tasks.basic.arm.reorient_sar",
        "myosuite.envs.tasks.arm.reorient_sar.env",
    ),
    (
        "myosuite.envs.myo.tasks.basic.arm.reorient_sar_geometries",
        "myosuite.envs.tasks.arm.reorient_sar.reorient_sar_geometries",
    ),
    (
        "myosuite.envs.myo.tasks.basic.hand.obj_hold",
        "myosuite.envs.tasks.hand.obj_hold.env",
    ),
    ("myosuite.envs.myo.tasks.basic.hand.pen", "myosuite.envs.tasks.hand.pen.env"),
    ("myosuite.envs.myo.tasks.basic.leg.reach", "myosuite.envs.tasks.leg.reach.env"),
    (
        "myosuite.envs.myo.tasks.basic.leg.specs.leg_directional_spec",
        "myosuite.envs.tasks.leg.directional.spec",
    ),
    ("myosuite.envs.myo.tasks.basic.leg.walk", "myosuite.envs.tasks.leg.walk.env"),
    (
        "myosuite.envs.myo.tasks.basic.muscle_mixin",
        "myosuite.envs.tasks.suites.muscle_mixin",
    ),
    (
        "myosuite.envs.myo.tasks.basic.specs.elbow_pose_spec",
        "myosuite.envs.tasks.arm.pose.spec",
    ),
    ("myosuite.envs.myo.tasks.basic.torso.pose", "myosuite.envs.tasks.torso.pose.env"),
    (
        "myosuite.envs.myo.tasks.challenge.baoding",
        "myosuite.envs.tasks.hand.baoding.env",
    ),
    (
        "myosuite.envs.myo.tasks.challenge.bimanual",
        "myosuite.envs.tasks.fullbody.bimanual.env",
    ),
    (
        "myosuite.envs.myo.tasks.challenge.challenge_common",
        "myosuite.envs.tasks.suites.challenge_common",
    ),
    (
        "myosuite.envs.myo.tasks.challenge.chase_tag_fb_model",
        "myosuite.envs.tasks.games.chasetag.chase_tag_fb_model",
    ),
    (
        "myosuite.envs.myo.tasks.challenge.chase_tag_vs.arena",
        "myosuite.envs.tasks.games.chasetag.chase_tag_vs.arena",
    ),
    (
        "myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_config",
        "myosuite.envs.tasks.games.chasetag.chase_tag_vs.chase_tag_vs_config",
    ),
    (
        "myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_model",
        "myosuite.envs.tasks.games.chasetag.chase_tag_vs.chase_tag_vs_model",
    ),
    (
        "myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_task_config",
        "myosuite.envs.tasks.games.chasetag.chase_tag_vs.chase_tag_vs_task_config",
    ),
    (
        "myosuite.envs.myo.tasks.challenge.chasetag",
        "myosuite.envs.tasks.games.chasetag.env",
    ),
    (
        "myosuite.envs.myo.tasks.challenge.combat_model_meta",
        "myosuite.envs.tasks.games.chasetag.combat_model_meta",
    ),
    (
        "myosuite.envs.myo.tasks.challenge.relocate",
        "myosuite.envs.tasks.fullbody.relocate.env",
    ),
    (
        "myosuite.envs.myo.tasks.challenge.reorient",
        "myosuite.envs.tasks.hand.reorient.env",
    ),
    (
        "myosuite.envs.myo.tasks.challenge.run_track",
        "myosuite.envs.tasks.leg.run_track.env",
    ),
    (
        "myosuite.envs.myo.tasks.challenge.soccer",
        "myosuite.envs.tasks.games.soccer.env",
    ),
    (
        "myosuite.envs.myo.tasks.challenge.tabletennis",
        "myosuite.envs.tasks.games.tabletennis.env",
    ),
    (
        "myosuite.envs.myo.tasks.challenge.tabletennis_spec",
        "myosuite.envs.tasks.games.tabletennis.spec",
    ),
    (
        "myosuite.envs.myo.tasks.mimic.chasetag_obs",
        "myosuite.envs.tasks.mimic.chasetag_obs",
    ),
    ("myosuite.envs.myo.tasks.mimic.clip_env", "myosuite.envs.tasks.mimic.clip_env"),
    ("myosuite.envs.myo.tasks.mimic.cpu", "myosuite.envs.tasks.mimic.cpu"),
    ("myosuite.envs.myo.tasks.mimic.policy", "myosuite.envs.tasks.mimic.policy"),
    (
        "myosuite.envs.myo.tasks.mimic.tracking.config",
        "myosuite.envs.tasks.mimic.tracking.config",
    ),
    (
        "myosuite.envs.myo.tasks.mimic.tracking.env",
        "myosuite.envs.tasks.mimic.tracking.env",
    ),
    (
        "myosuite.envs.myo.tasks.mimic.tracking.metrics",
        "myosuite.envs.tasks.mimic.tracking.metrics",
    ),
    (
        "myosuite.envs.myo.tasks.mimic.tracking.models",
        "myosuite.envs.tasks.mimic.tracking.models",
    ),
    (
        "myosuite.envs.myo.tasks.mimic.tracking.motion",
        "myosuite.envs.tasks.mimic.tracking.motion",
    ),
    (
        "myosuite.envs.myo.tasks.mimic.tracking.sampling",
        "myosuite.envs.tasks.mimic.tracking.sampling",
    ),
)

_COMPATIBILITY_PACKAGE_FACADES: tuple[tuple[str, str], ...] = (
    ("myosuite.envs.myo.backends", "myosuite.envs.backends"),
    ("myosuite.envs.myo.backends.mjlab", "myosuite.envs.backends.mjlab"),
    (
        "myosuite.envs.myo.backends.mjlab.configs",
        "myosuite.envs.backends.mjlab.configs",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.configs.asset_configs",
        "myosuite.envs.backends.mjlab.configs.asset_configs",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.configs.asset_configs.muscle_mimic_body",
        "myosuite.envs.backends.mjlab.configs.asset_configs.muscle_mimic_body",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.configs.task_configs",
        "myosuite.envs.backends.mjlab.configs.task_configs",
    ),
    ("myosuite.envs.myo.backends.mjlab.tasks", "myosuite.envs.backends.mjlab.tasks"),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.leg",
        "myosuite.envs.tasks.leg.walk.mjlab_register",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.mdp",
        "myosuite.envs.backends.mjlab.tasks.mdp",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.mimic_tracking",
        "myosuite.envs.tasks.mimic.mjlab_register",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.pose",
        "myosuite.envs.backends.mjlab.tasks.pose",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.pose.config.elbow",
        "myosuite.envs.tasks.arm.pose.mjlab_register",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.pose.config.finger",
        "myosuite.envs.tasks.hand.pose.finger_mjlab_register",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.pose.config.hand",
        "myosuite.envs.tasks.hand.pose.hand_mjlab_register",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.pose.config.torso",
        "myosuite.envs.tasks.torso.pose.mjlab_register",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.pose.mdp",
        "myosuite.envs.backends.mjlab.tasks.pose.mdp",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.reach",
        "myosuite.envs.backends.mjlab.tasks.reach",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.reach.config.arm",
        "myosuite.envs.tasks.arm.reach.mjlab_register",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.reach.config.finger",
        "myosuite.envs.tasks.hand.reach.finger_mjlab_register",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.reach.config.hand",
        "myosuite.envs.tasks.hand.reach.hand_mjlab_register",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.reach.mdp",
        "myosuite.envs.backends.mjlab.tasks.reach.mdp",
    ),
    (
        "myosuite.envs.myo.backends.mjlab.tasks.waypoint",
        "myosuite.envs.tasks.fullbody.waypoint.mjlab_register",
    ),
    ("myosuite.envs.myo.backends.mjx", "myosuite.envs.backends._experimental_mjx"),
    ("myosuite.envs.myo.myoedits", "myosuite.envs.tasks.suites.edits"),
    ("myosuite.envs.myo.tasks.basic", "myosuite.envs.tasks.suites.basic"),
    ("myosuite.envs.myo.tasks.challenge", "myosuite.envs.tasks.suites.challenge"),
    (
        "myosuite.envs.myo.tasks.challenge.chase_tag_vs",
        "myosuite.envs.tasks.games.chasetag.chase_tag_vs",
    ),
    ("myosuite.envs.myo.tasks.mimic", "myosuite.envs.tasks.mimic"),
    ("myosuite.envs.myo.tasks.mimic.tracking", "myosuite.envs.tasks.mimic.tracking"),
)

_NON_SHIM_FACADE_TARGETS: dict[str, tuple[str, ...]] = {
    "envs/myo/__init__.py": (),
    "envs/myo/assets/__init__.py": (),
    "envs/myo/assets/leg/__init__.py": (),
    "envs/myo/backends/mjlab/tasks/pose/config/__init__.py": (
        "myosuite.envs.myo.backends.mjlab.tasks.pose.config",
    ),
    "envs/myo/backends/mjlab/tasks/reach/config/__init__.py": (
        "myosuite.envs.myo.backends.mjlab.tasks.reach.config",
    ),
    "envs/myo/tasks/__init__.py": (),
    "envs/myo/tasks/basic/arm/__init__.py": (),
    "envs/myo/tasks/basic/full_body/__init__.py": (),
    "envs/myo/tasks/basic/hand/__init__.py": (),
    "envs/myo/tasks/basic/leg/__init__.py": (),
    "envs/myo/tasks/basic/leg/specs/__init__.py": (
        "myosuite.envs.tasks.leg.directional.spec",
    ),
    "envs/myo/tasks/basic/specs/__init__.py": (
        "myosuite.envs.tasks.arm.pose.spec",
        "myosuite.envs.tasks.leg.directional.spec",
    ),
    "envs/myo/tasks/basic/torso/__init__.py": (),
    "physics/inverse_kinematics.py": ("myosuite.biomechanics.inverse_kinematics",),
    "scenes/__init__.py": (
        "myosuite.scenes",
        "myosuite.core.builder.scene_library",
    ),
    "terms/__init__.py": ("myosuite.core.terms", "myosuite.terms"),
    "terms/multiplayer/__init__.py": (),
    "logger/__init__.py": (),
    "utils/__init__.py": ("myosuite.core.seeding",),
    "utils/gym.py": (),
}

_COMPATIBILITY_FACADE_FILES: frozenset[str] = frozenset(
    {
        *(
            f"{legacy.removeprefix('myosuite.').replace('.', '/')}.py"
            for legacy, _ in _COMPATIBILITY_MODULE_SHIMS
        ),
        *(
            f"{legacy.removeprefix('myosuite.').replace('.', '/')}/__init__.py"
            for legacy, _ in _COMPATIBILITY_PACKAGE_FACADES
        ),
        *_NON_SHIM_FACADE_TARGETS,
    }
)

_OPTIONAL_TOP_LEVEL_DEPS = frozenset(
    {"brax", "etils", "flax", "jax", "mink", "mjlab", "onnxruntime", "torch"}
)


def _runtime_python_files() -> list[Path]:
    """Return runtime ``.py`` files inside ``myosuite/`` (excluding ``tests/`` and ``agents/``)."""
    proc = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "myosuite",
        ],
        cwd=_REPO,
        check=False,
        capture_output=True,
    )
    if proc.returncode == 0:
        candidates = [
            _REPO / raw
            for raw in proc.stdout.decode("utf-8").split("\0")
            if raw and raw.endswith(".py")
        ]
    else:
        candidates = [p for p in _PACKAGE.rglob("*.py") if "__pycache__" not in p.parts]
    paths: list[Path] = []
    for candidate in candidates:
        rel = candidate.relative_to(_REPO).as_posix()
        if rel.startswith("myosuite/tests/") or rel.startswith("myosuite/agents/"):
            continue
        if candidate.is_file():
            paths.append(candidate)
    return sorted(set(paths))


def _is_type_checking_guard(test: ast.expr) -> bool:
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if (
        isinstance(test, ast.Attribute)
        and test.attr == "TYPE_CHECKING"
        and isinstance(test.value, ast.Name)
        and test.value.id in {"typing", "t"}
    ):
        return True
    return False


def _resolve_relative_module(
    rel_parts: list[str], is_pkg: bool, name: str, level: int
) -> str:
    anchor_parts = ["myosuite", *rel_parts] if is_pkg else ["myosuite", *rel_parts[:-1]]
    keep = len(anchor_parts) - (level - 1)
    prefix = ".".join(anchor_parts[: max(keep, 0)])
    return f"{prefix}.{name}" if (prefix and name) else (name or prefix)


def _is_import_module_call(func: ast.expr) -> bool:
    if isinstance(func, ast.Name) and func.id == "import_module":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "import_module":
        return True
    return False


def _runtime_imports(
    py_file: Path, *, include_type_checking: bool = False
) -> list[str]:
    """Extract import targets (including literal ``import_module(...)`` calls)."""
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(py_file))

    type_checking_nodes: set[ast.AST] = set()
    if not include_type_checking:
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and _is_type_checking_guard(node.test):
                for stmt in node.body:
                    type_checking_nodes.update(ast.walk(stmt))

    rel_parts = list(py_file.relative_to(_PACKAGE).with_suffix("").parts)
    is_pkg = rel_parts[-1] == "__init__"
    if is_pkg:
        rel_parts = rel_parts[:-1]

    imports: list[str] = []
    for node in ast.walk(tree):
        if node in type_checking_nodes:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    if node.module == "myosuite":
                        imports.append(node.module)
                        for alias in node.names:
                            if alias.name != "*":
                                imports.append(f"myosuite.{alias.name}")
                    elif node.module.startswith("myosuite."):
                        if any(a.name == "*" for a in node.names):
                            imports.append(node.module)
                        mod_rel = Path(
                            *node.module.removeprefix("myosuite.").split(".")
                        )
                        for alias in node.names:
                            if alias.name == "*":
                                continue
                            sub_py = _PACKAGE / mod_rel / f"{alias.name}.py"
                            sub_pkg = _PACKAGE / mod_rel / alias.name / "__init__.py"
                            if sub_py.is_file() or sub_pkg.is_file():
                                imports.append(f"{node.module}.{alias.name}")
                            else:
                                imports.append(node.module)
                                imports.append(f"{node.module}.{alias.name}")
                    else:
                        imports.append(node.module)
            else:
                base_mod = _resolve_relative_module(
                    rel_parts, is_pkg, node.module, node.level
                )
                if node.module:
                    imports.append(base_mod)
                for alias in node.names:
                    if alias.name != "*":
                        imports.append(
                            f"{base_mod}.{alias.name}" if base_mod else alias.name
                        )
        elif (
            isinstance(node, ast.Call)
            and _is_import_module_call(node.func)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            raw_mod = node.args[0].value
            pkg_arg: str | None = None
            if (
                len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                pkg_arg = node.args[1].value
            for kw in node.keywords:
                if (
                    kw.arg == "package"
                    and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)
                ):
                    pkg_arg = kw.value.value
            if raw_mod.startswith("."):
                level = len(raw_mod) - len(raw_mod.lstrip("."))
                remainder = raw_mod[level:]
                if pkg_arg:
                    pkg_parts = pkg_arg.split(".")
                    keep = len(pkg_parts) - (level - 1)
                    prefix = ".".join(pkg_parts[: max(keep, 0)])
                    imports.append(
                        f"{prefix}.{remainder}"
                        if (prefix and remainder)
                        else (remainder or prefix)
                    )
                else:
                    imports.append(
                        _resolve_relative_module(rel_parts, is_pkg, remainder, level)
                    )
            else:
                imports.append(raw_mod)
        elif (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(t, ast.Name)
                and t.id in {"_LAZY_EXPORTS", "_COMPAT_SUBMODULES"}
                for t in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
            )
            and node.value is not None
        ):
            for sub in ast.walk(node.value):
                if (
                    isinstance(sub, ast.Constant)
                    and isinstance(sub.value, str)
                    and sub.value.startswith("myosuite.")
                ):
                    imports.append(sub.value)
    return imports


def test_approved_myosuite_subpackages() -> None:
    """Every tracked top-level subdirectory of ``myosuite/`` must be an approved subpackage."""
    found = {
        path.relative_to(_PACKAGE).parts[0]
        for path in _runtime_python_files()
        if len(path.relative_to(_PACKAGE).parts) > 1
    }
    unexpected = sorted(found - _APPROVED_RUNTIME_SUBPACKAGES)
    assert not unexpected, (
        f"Unapproved top-level myosuite/ subpackages: {unexpected}. "
        f"Approved: {sorted(_APPROVED_RUNTIME_SUBPACKAGES)}"
    )


def test_myosuite_never_imports_out_of_wheel_or_test_packages() -> None:
    """Runtime modules inside ``myosuite/`` must never import ``training``, ``tools``, ``scripts``, or ``tests``."""
    violations: list[str] = []
    for path in _runtime_python_files():
        rel = path.relative_to(_PACKAGE).as_posix()
        for mod in _runtime_imports(path):
            top = mod.split(".")[0]
            if top in _FORBIDDEN_EXTERNAL_IMPORTS:
                violations.append(f"{rel} -> {mod}")
            elif mod == "myosuite.tests" or mod.startswith("myosuite.tests."):
                violations.append(f"{rel} -> {mod}")
    assert not violations, (
        "Runtime myosuite modules must not import out-of-wheel or test packages:\n  "
        + "\n  ".join(violations)
    )


def _matches_allowed_prefix(mod: str, prefixes: tuple[str, ...]) -> bool:
    return any(mod == p or mod.startswith(f"{p}.") for p in prefixes)


def test_myosuite_internal_import_dag() -> None:
    """Enforce the upward module import DAG across ``myosuite/``."""
    for rel_file, dst_sub in _LEGACY_CROSS_IMPORTS:
        src_sub = rel_file.split("/", 1)[0]
        assert src_sub not in _ZERO_EXEMPTION_SUBPACKAGES, (
            f"Subpackage {src_sub!r} must have zero outgoing legacy exemptions: "
            f"({rel_file!r}, {dst_sub!r})"
        )

    shim_map = dict(_COMPATIBILITY_MODULE_SHIMS)
    facade_allowed_prefixes: dict[str, tuple[str, ...]] = {
        **{
            f"{legacy.removeprefix('myosuite.').replace('.', '/')}.py": (canonical,)
            for legacy, canonical in _COMPATIBILITY_MODULE_SHIMS
        },
        **{
            f"{legacy.removeprefix('myosuite.').replace('.', '/')}/__init__.py": (
                canonical,
                legacy,
                *(
                    ("myosuite.envs.backends.mjlab.bootstrap",)
                    if legacy == "myosuite.envs.myo.backends.mjlab.tasks"
                    else ()
                ),
            )
            for legacy, canonical in _COMPATIBILITY_PACKAGE_FACADES
        },
        **_NON_SHIM_FACADE_TARGETS,
    }
    assert set(facade_allowed_prefixes) == _COMPATIBILITY_FACADE_FILES

    facade_violations: list[str] = []
    shim_imports: list[str] = []
    violations: set[tuple[str, str]] = set()
    for path in _runtime_python_files():
        rel = path.relative_to(_PACKAGE)
        rel_posix = rel.as_posix()
        if rel_posix in _COMPATIBILITY_FACADE_FILES:
            allowed_prefixes = facade_allowed_prefixes[rel_posix]
            for mod in sorted(set(_runtime_imports(path, include_type_checking=True))):
                if mod == "myosuite" or mod.startswith("myosuite."):
                    if not _matches_allowed_prefix(mod, allowed_prefixes):
                        facade_violations.append(
                            f"{rel_posix} -> {mod} (allowed: {allowed_prefixes})"
                        )
            continue
        src_sub = rel.parts[0] if len(rel.parts) > 1 else "__root__"
        allowed = _ALLOWED_INTERNAL_IMPORTS[src_sub]
        for mod in _runtime_imports(path):
            if mod in shim_map:
                shim_imports.append(f"{rel_posix} -> {mod} (use {shim_map[mod]})")
            mod = shim_map.get(mod, mod)
            parts = mod.split(".")
            if parts[0] != "myosuite" or len(parts) < 2:
                continue
            dst_sub = (
                parts[1] if parts[1] in _APPROVED_RUNTIME_SUBPACKAGES else "__root__"
            )
            if dst_sub not in allowed:
                violations.add((rel.as_posix(), dst_sub))

    assert not facade_violations, (
        "Compatibility facade files may only import their mapped canonical target(s):\n  "
        + "\n  ".join(facade_violations)
    )
    assert not shim_imports, (
        "Non-facade runtime modules must import canonical paths rather than compatibility shims:\n  "
        + "\n  ".join(sorted(shim_imports))
    )
    known_exemptions = _LEGACY_CROSS_IMPORTS | _COMPATIBILITY_REEXPORT_IMPORTS
    unexpected = sorted(violations - known_exemptions)
    assert not unexpected, (
        "Forbidden cross-subpackage runtime imports detected (see docs/wiki/repository-map.md):\n  "
        + "\n  ".join(f"{src} -> myosuite.{dst}" for src, dst in unexpected)
    )
    stale = sorted(known_exemptions - violations)
    assert not stale, (
        "Stale _LEGACY_CROSS_IMPORTS entries no longer occur — remove them so fixed edges cannot regress:\n  "
        + "\n  ".join(f"{src} -> myosuite.{dst}" for src, dst in stale)
    )


def test_runtime_imports_checks_type_checking_else_and_import_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_runtime_imports`` must check ``if TYPE_CHECKING: ... else:`` runtime branches and literal ``import_module`` calls."""
    pkg = tmp_path / "myosuite"
    (pkg / "core").mkdir(parents=True)
    sample = pkg / "core" / "sample.py"
    sample.write_text(
        "from typing import TYPE_CHECKING\n"
        "import importlib\n"
        "if TYPE_CHECKING:\n"
        "    import myosuite.viz\n"
        "else:\n"
        "    import myosuite.envs\n"
        "from myosuite import logger, make_env\n"
        "from myosuite.terms import mimic_track\n"
        "from ..utils import asset_path_resolver\n"
        "importlib.import_module('myosuite.integrations')\n"
        "importlib.import_module('.submod', package='myosuite.biomechanics')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("myosuite.tests.test_package_boundaries._PACKAGE", pkg)
    found = set(_runtime_imports(sample))
    assert "myosuite.viz" not in found
    assert "myosuite.envs" in found
    assert "myosuite.logger" in found
    assert "myosuite.make_env" in found
    assert "myosuite.terms.mimic_track" in found
    assert "myosuite.utils.asset_path_resolver" in found
    assert "myosuite.integrations" in found
    assert "myosuite.biomechanics.submod" in found


def test_compatibility_shims_are_pure_aliases() -> None:
    """Every migrated leaf shim resolves to the identical module object, and package facades preserve __file__/resources."""
    import importlib
    import importlib.resources as resources
    import runpy
    import sys
    import types

    for legacy_mod, canonical_mod in _COMPATIBILITY_MODULE_SHIMS:
        shim_file = _REPO / f"{legacy_mod.replace('.', '/')}.py"
        assert shim_file.is_file(), f"Missing shim file: {shim_file}"
        try:
            legacy = importlib.import_module(legacy_mod)
            canonical = importlib.import_module(canonical_mod)
        except ModuleNotFoundError as exc:
            missing_top = (exc.name or "").split(".")[0]
            if missing_top in _OPTIONAL_TOP_LEVEL_DEPS:
                continue
            raise
        assert "TYPE_CHECKING" in shim_file.read_text(
            encoding="utf-8"
        ), f"Leaf shim {shim_file} must expose TYPE_CHECKING star-imports"
        assert (
            legacy is canonical
        ), f"{legacy_mod} ({legacy!r}) is not identical to {canonical_mod} ({canonical!r})"

    for legacy_pkg_name, canonical_mod_name in _COMPATIBILITY_PACKAGE_FACADES:
        pkg_init = _REPO / f"{legacy_pkg_name.replace('.', '/')}/__init__.py"
        assert pkg_init.is_file(), f"Missing package facade: {pkg_init}"
        try:
            legacy_pkg = importlib.import_module(legacy_pkg_name)
            canonical_mod = importlib.import_module(canonical_mod_name)
        except ModuleNotFoundError as exc:
            missing_top = (exc.name or "").split(".")[0]
            if missing_top in _OPTIONAL_TOP_LEVEL_DEPS:
                continue
            raise
        assert legacy_pkg.__file__ is not None and Path(legacy_pkg.__file__).is_file()
        for attr, val in canonical_mod.__dict__.items():
            if not attr.startswith("__") and not isinstance(val, types.ModuleType):
                assert getattr(legacy_pkg, attr) is val

    # Unaliased legacy packages and resource lookups continue to work natively.
    myo_pkg = importlib.import_module("myosuite.envs.myo")
    assert myo_pkg.__file__ is not None and Path(myo_pkg.__file__).is_file()
    assert resources.files("myosuite.envs.myo").is_dir()
    fb_pkg = importlib.import_module("myosuite.envs.myo.tasks.basic.full_body")
    assert fb_pkg.__file__ is not None and Path(fb_pkg.__file__).is_file()
    assert (resources.files("myosuite.envs.myo.assets") / "myo_relief.npy").is_file()
    assert (
        resources.files("myosuite.envs.myo.assets") / "arm" / "myoarm_relocate.xml"
    ).is_file()

    for closed_dir in ("scenes", "terms", "logger", "envs/myo"):
        on_disk = {
            p.relative_to(_PACKAGE).as_posix()
            for p in (_PACKAGE / closed_dir).rglob("*.py")
        }
        unlisted = sorted(on_disk - _COMPATIBILITY_FACADE_FILES)
        assert (
            not unlisted
        ), f"Closed namespace myosuite/{closed_dir}/ has non-facade files: {unlisted}"

    utils_on_disk = {
        p.relative_to(_PACKAGE).as_posix() for p in (_PACKAGE / "utils").rglob("*.py")
    }
    utils_allowed = _COMPATIBILITY_FACADE_FILES | {
        "utils/sb3_callbacks.py",
        "utils/sb3_sweep.py",
    }
    utils_unlisted = sorted(utils_on_disk - utils_allowed)
    assert (
        not utils_unlisted
    ), f"Closed namespace myosuite/utils/ has unexpected files: {utils_unlisted}"

    for rel_facade in sorted(_COMPATIBILITY_FACADE_FILES):
        facade_file = _PACKAGE / rel_facade
        assert facade_file.is_file(), f"Missing facade file: {facade_file}"
        tree = ast.parse(
            facade_file.read_text(encoding="utf-8"), filename=str(facade_file)
        )
        defs = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef))
            or (isinstance(node, ast.FunctionDef) and node.name != "__getattr__")
        ]
        assert (
            not defs
        ), f"Compatibility facade {facade_file} must not define new logic: {defs}"

    for runnable_legacy in (
        "myosuite.envs.myo.backends.mjx.train_jax_ppo",
        "myosuite.envs.myo.backends.mjx.visu_mjx_env",
        "myosuite.envs.myo.backends.mjlab.configs.asset_configs.muscle_mimic_body.full_body_constants",
    ):
        sys.modules.pop(runnable_legacy, None)
        try:
            _, spec, code = runpy._get_module_details(runnable_legacy)
        except ModuleNotFoundError as exc:
            missing_top = (exc.name or "").split(".")[0]
            if missing_top in _OPTIONAL_TOP_LEVEL_DEPS:
                continue
            raise
        assert spec is not None and code is not None


def test_scenes_package_binds_library_submodule_in_fresh_process() -> None:
    """Legacy package attribute access (``myosuite.scenes.library``, ``myosuite.terms.*``, ``myosuite.utils.*``) works in a fresh interpreter."""
    code = (
        "import myosuite, myosuite.scenes as s; "
        "assert hasattr(s, 'library') and s.library.SCENES; "
        "assert myosuite.utils.spec_processing is not None; "
        "assert myosuite.utils.asset_path_resolver is not None; "
        "assert myosuite.terms.waypoint is not None; "
        "assert myosuite.terms.opponent_relative_obs is not None; "
        "assert myosuite.terms.multiplayer.heading is not None"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_legacy_basic_and_challenge_facades_bind_submodules_in_fresh_process() -> None:
    """Legacy ``basic`` and ``challenge`` package facades still expose child submodules like ``c.chasetag`` and ``b.specs``."""
    code = (
        "import myosuite.envs.myo.tasks.basic as b; "
        "import myosuite.envs.myo.tasks.challenge as c; "
        "assert hasattr(b, 'specs') and hasattr(b, 'muscle_mixin'); "
        "assert hasattr(c, 'chasetag') and hasattr(c.chasetag, 'ChaseTagEnv')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
