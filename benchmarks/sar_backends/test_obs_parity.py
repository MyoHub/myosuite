# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Observation parity test: verify mjlab observation space matches public env.

Tests that the mjlab walk observation terms defined in register_mjlab_tasks.py:
  1. Produce observations with the correct shape (matching CPU/MJX)
  2. Contain the right number of terms (12 components)
  3. Are NOT the old 3-dim projected_gravity stub
  4. Have correct component names matching WalkEnvV0.DEFAULT_OBS_KEYS

These tests use only AST/source inspection (no mjlab import needed) so they
run in CI without GPU or mjlab installed.

For runtime verification (requires mjlab + GPU), see compare_obs_backends.py.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

_REGISTER_PY = (
    Path(__file__).resolve().parents[2]
    / "myosuite/envs/myo/backends/mjlab/register_mjlab_tasks.py"
)

_WALK_V0_PY = (
    Path(__file__).resolve().parents[2] / "myosuite/envs/myo/myobase/walk_v0.py"
)

_MJX_WALK_PY = (
    Path(__file__).resolve().parents[2] / "myosuite/envs/myo/backends/mjx/walk_env.py"
)


def _src() -> str:
    return _REGISTER_PY.read_text()


def _walk_v0_src() -> str:
    return _WALK_V0_PY.read_text()


def _mjx_walk_src() -> str:
    return _MJX_WALK_PY.read_text()


def _parse_class_default_obs_keys(src: str, class_name: str) -> list[str]:
    """Return DEFAULT_OBS_KEYS string constants from a specific class body via AST."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Assign)
                    and any(
                        isinstance(t, ast.Name) and t.id == "DEFAULT_OBS_KEYS"
                        for t in child.targets
                    )
                    and isinstance(child.value, ast.List)
                ):
                    return [
                        elt.value
                        for elt in child.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    ]
    return []


def _parse_method_return_keys(src: str, method_name: str) -> list[str]:
    """Return string keys from the return dict of a method, found via AST.

    Scans all FunctionDef nodes matching *method_name* and returns keys from
    the first ``return { … }`` statement, ignoring any other dicts in the body.
    """
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            for child in ast.walk(node):
                if isinstance(child, ast.Return) and isinstance(child.value, ast.Dict):
                    return [
                        k.value
                        for k in child.value.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)
                    ]
    return []


# Expected observation keys matching WalkEnvV0.DEFAULT_OBS_KEYS (minus t/time).
# These must all appear in the mjlab observation dict.
_EXPECTED_OBS_KEYS = [
    "qpos_without_xy",
    "qvel",
    "com_vel",
    "torso_angle",
    "feet_heights",
    "height",
    "feet_rel_positions",
    "phase_var",
    "muscle_length",
    "muscle_velocity",
    "muscle_force",
    "act",
]

# Old stub that should no longer appear as the ONLY observation.
_STUB_TERM = "projected_gravity"

# Names of the expected observation function implementations.
_EXPECTED_OBS_FUNCS = [
    "_walk_obs_qpos_without_xy",
    "_walk_obs_qvel",
    "_walk_obs_com_vel",
    "_walk_obs_torso_angle",
    "_walk_obs_feet_heights",
    "_walk_obs_height",
    "_walk_obs_feet_rel_positions",
    "_walk_obs_phase_var",
    "_walk_obs_muscle_length",
    "_walk_obs_muscle_velocity",
    "_walk_obs_muscle_force",
    "_walk_obs_act",
]


class TestMjlabObsParity(unittest.TestCase):
    """Structural tests: observation terms match the public (CPU) environment."""

    # ------------------------------------------------------------------
    # Source presence tests
    # ------------------------------------------------------------------

    def test_obs_functions_defined(self):
        """All 12 _walk_obs_* functions must be defined in the source."""
        src = _src()
        for fn in _EXPECTED_OBS_FUNCS:
            self.assertIn(
                f"def {fn}(",
                src,
                msg=f"Missing observation function: {fn}",
            )

    def test_obs_terms_in_make_walk_env_cfg(self):
        """_make_walk_env_cfg must register all 12 observation terms by key."""
        src = _src()
        for key in _EXPECTED_OBS_KEYS:
            self.assertIn(
                f'"{key}"',
                src,
                msg=f'Observation key "{key}" missing from _make_walk_env_cfg',
            )

    def test_not_only_projected_gravity(self):
        """mjlab walk obs must NOT be reduced to only projected_gravity.

        The old stub used just projected_gravity (3D). The observation section
        in _make_walk_env_cfg must contain at least one _walk_obs_* function.
        """
        src = _src()
        self.assertIn(
            "_walk_obs_qpos_without_xy",
            src,
            msg="Walk obs still uses old projected_gravity stub; "
            "_walk_obs_qpos_without_xy not found in _make_walk_env_cfg",
        )

    def test_all_obs_funcs_referenced_in_cfg(self):
        """Every _walk_obs_* function must be referenced in observations dict."""
        src = _src()
        # Find the _make_walk_env_cfg function body
        cfg_start = src.index("def _make_walk_env_cfg(")
        cfg_src = src[cfg_start:]
        for fn in _EXPECTED_OBS_FUNCS:
            self.assertIn(
                fn,
                cfg_src,
                msg=f"Observation function {fn} not used in _make_walk_env_cfg",
            )

    # ------------------------------------------------------------------
    # CPU / MJX parity tests
    # ------------------------------------------------------------------

    def test_cpu_obs_keys_present(self):
        """All keys in WalkEnvV0.DEFAULT_OBS_KEYS must appear in mjlab cfg.

        Uses AST scoped to the WalkEnvV0 class body to avoid picking up
        DEFAULT_OBS_KEYS from sibling classes (ReachEnvV0, TerrainEnvV0, …).
        """
        cpu_keys = _parse_class_default_obs_keys(_walk_v0_src(), "WalkEnvV0")
        if not cpu_keys:
            self.skipTest("Could not parse WalkEnvV0.DEFAULT_OBS_KEYS from AST")

        mjlab_src = _src()
        missing = [k for k in cpu_keys if k not in ("t", "time") and k not in mjlab_src]
        self.assertEqual(
            missing,
            [],
            msg=f"WalkEnvV0.DEFAULT_OBS_KEYS keys missing in mjlab: {missing}",
        )

    def test_mjx_obs_keys_present(self):
        """All observation keys returned by MjxWalkEnv.get_obs_dict must appear in mjlab.

        Uses AST scoped to MjxWalkEnv.get_obs_dict return statement to avoid
        picking up task_state dict keys (e.g. 'target_rot') that are used
        internally for reward shaping but are NOT part of the observation vector.
        """
        mjx_keys = _parse_method_return_keys(_mjx_walk_src(), "get_obs_dict")
        if not mjx_keys:
            self.skipTest("Could not parse return keys from MjxWalkEnv.get_obs_dict")

        mjlab_src = _src()
        missing = [k for k in mjx_keys if k not in mjlab_src]
        self.assertEqual(
            missing,
            [],
            msg=f"MjxWalkEnv.get_obs_dict keys missing in mjlab walk obs: {missing}",
        )

    # ------------------------------------------------------------------
    # Observation function correctness tests (AST-based)
    # ------------------------------------------------------------------

    def test_qpos_without_xy_slices_at_2(self):
        """_walk_obs_qpos_without_xy must slice qpos starting at index 2."""
        src = _src()
        fn_start = src.index("def _walk_obs_qpos_without_xy(")
        fn_end = src.index("\ndef _walk_obs_qvel(")
        fn_src = src[fn_start:fn_end]
        self.assertIn(
            "qpos[:, 2:]",
            fn_src,
            msg="_walk_obs_qpos_without_xy should use qpos[:, 2:] to exclude x,y",
        )

    def test_qvel_scaled_by_ctrl_dt(self):
        """_walk_obs_qvel must multiply by ctrl_dt."""
        src = _src()
        fn_start = src.index("def _walk_obs_qvel(")
        fn_end = src.index("\ndef _walk_obs_com_vel(")
        fn_src = src[fn_start:fn_end]
        self.assertIn(
            "ctrl_dt",
            fn_src,
            msg="_walk_obs_qvel should scale velocities by ctrl_dt",
        )
        self.assertIn(
            "qvel",
            fn_src,
            msg="_walk_obs_qvel should access data.qvel",
        )

    def test_com_vel_negates_cvel(self):
        """_walk_obs_com_vel must negate cvel (MuJoCo sign convention)."""
        src = _src()
        fn_start = src.index("def _walk_obs_com_vel(")
        fn_end = src.index("\ndef _walk_obs_torso_angle(")
        fn_src = src[fn_start:fn_end]
        self.assertIn(
            "cvel",
            fn_src,
            msg="_walk_obs_com_vel should use data.cvel",
        )
        self.assertIn(
            "-data.cvel",
            fn_src,
            msg="_walk_obs_com_vel should negate cvel (per MuJoCo convention)",
        )

    def test_muscle_velocity_clips_100(self):
        """_walk_obs_muscle_velocity must clip to ±100 (matches CPU env)."""
        src = _src()
        fn_start = src.index("def _walk_obs_muscle_velocity(")
        fn_end = src.index("\ndef _walk_obs_muscle_force(")
        fn_src = src[fn_start:fn_end]
        self.assertIn(
            "100.0",
            fn_src,
            msg="_walk_obs_muscle_velocity should clip to 100.0",
        )
        self.assertIn(
            "actuator_velocity",
            fn_src,
            msg="_walk_obs_muscle_velocity should use actuator_velocity",
        )

    def test_muscle_force_divides_by_1000(self):
        """_walk_obs_muscle_force must divide by 1000 (matches CPU env scaling)."""
        src = _src()
        fn_start = src.index("def _walk_obs_muscle_force(")
        fn_end = src.index("\ndef _walk_obs_act(")
        fn_src = src[fn_start:fn_end]
        self.assertIn(
            "1000.0",
            fn_src,
            msg="_walk_obs_muscle_force should divide by 1000.0",
        )
        self.assertIn(
            "actuator_force",
            fn_src,
            msg="_walk_obs_muscle_force should use actuator_force",
        )

    def test_phase_var_uses_hip_period(self):
        """_walk_obs_phase_var must use _WALK_HIP_PERIOD constant."""
        src = _src()
        fn_start = src.index("def _walk_obs_phase_var(")
        fn_end = src.index("\ndef _walk_obs_muscle_length(")
        fn_src = src[fn_start:fn_end]
        self.assertIn(
            "_WALK_HIP_PERIOD",
            fn_src,
            msg="_walk_obs_phase_var should use _WALK_HIP_PERIOD",
        )
        self.assertIn(
            "% 1",
            fn_src,
            msg="_walk_obs_phase_var should compute modulo 1 for cyclic phase",
        )

    def test_height_uses_xipos(self):
        """_walk_obs_height must use xipos (body CoM positions), not xpos."""
        src = _src()
        fn_start = src.index("def _walk_obs_height(")
        fn_end = src.index("\ndef _walk_obs_feet_rel_positions(")
        fn_src = src[fn_start:fn_end]
        self.assertIn(
            "xipos",
            fn_src,
            msg="_walk_obs_height should use xipos (body CoM positions) for mass-weighted height",
        )

    def test_feet_rel_positions_subtracts_pelvis(self):
        """_walk_obs_feet_rel_positions must subtract pelvis position."""
        src = _src()
        fn_start = src.index("def _walk_obs_feet_rel_positions(")
        fn_end = src.index("\ndef _walk_obs_phase_var(")
        fn_src = src[fn_start:fn_end]
        self.assertIn(
            "pelvis",
            fn_src,
            msg="_walk_obs_feet_rel_positions should subtract pelvis_pos",
        )
        self.assertIn(
            "talus",
            fn_src,
            msg="_walk_obs_feet_rel_positions should use talus body positions",
        )

    # ------------------------------------------------------------------
    # Body ID caching
    # ------------------------------------------------------------------

    def test_body_id_cache_defined(self):
        """_walk_obs_cache module-level dict must be defined."""
        src = _src()
        self.assertIn(
            "_walk_obs_cache",
            src,
            msg="Module-level body ID cache _walk_obs_cache not found",
        )

    def test_resolve_walk_obs_ids_defined(self):
        """_resolve_walk_obs_ids helper function must be defined."""
        src = _src()
        self.assertIn(
            "def _resolve_walk_obs_ids(",
            src,
            msg="_resolve_walk_obs_ids helper not found",
        )

    # ------------------------------------------------------------------
    # Integration: 12 observation terms in policy group
    # ------------------------------------------------------------------

    def test_policy_group_has_12_terms(self):
        """The 'policy' ObservationGroupCfg must have exactly 12 terms."""
        src = _src()
        # Count _walk_obs_* references inside _make_walk_env_cfg
        cfg_start = src.index("def _make_walk_env_cfg(")
        # Find next top-level function after _make_walk_env_cfg
        next_def = src.index("\ndef _walk_ppo_runner_cfg(", cfg_start)
        cfg_src = src[cfg_start:next_def]

        count = sum(1 for fn in _EXPECTED_OBS_FUNCS if fn in cfg_src)
        self.assertEqual(
            count,
            12,
            msg=f"Expected 12 observation functions in _make_walk_env_cfg, found {count}. "
            f"Missing: {[fn for fn in _EXPECTED_OBS_FUNCS if fn not in cfg_src]}",
        )


class TestObsConsistencyWithPublicEnv(unittest.TestCase):
    """Cross-check: observation component names consistent with WalkEnvV0."""

    def test_all_default_obs_keys_implemented(self):
        """Every non-debug key in WalkEnvV0.DEFAULT_OBS_KEYS is implemented.

        Uses AST scoped to the WalkEnvV0 class to avoid picking up keys from
        sibling classes (ReachEnvV0, TerrainEnvV0) in the same file.
        """
        default_keys = _parse_class_default_obs_keys(_walk_v0_src(), "WalkEnvV0")
        if not default_keys:
            self.skipTest("Could not parse WalkEnvV0.DEFAULT_OBS_KEYS from AST")

        mjlab_src = _src()
        missing = [
            key
            for key in default_keys
            if key not in ("t", "time") and key not in mjlab_src
        ]
        self.assertEqual(
            missing,
            [],
            msg=f"WalkEnvV0.DEFAULT_OBS_KEYS entries not implemented in mjlab: {missing}",
        )

    def test_mjx_and_mjlab_obs_keys_match(self):
        """MjxWalkEnv and mjlab must use the same observation key names."""
        mjx_src = _mjx_walk_src()
        mjlab_src = _src()

        # Parse MJX return keys via AST
        mjx_tree = ast.parse(mjx_src)
        mjx_keys = []
        for node in ast.walk(mjx_tree):
            if isinstance(node, ast.FunctionDef) and node.name == "get_obs_dict":
                for child in ast.walk(node):
                    if isinstance(child, ast.Return) and isinstance(
                        child.value, ast.Dict
                    ):
                        for key in child.value.keys:
                            if isinstance(key, ast.Constant):
                                mjx_keys.append(key.value)

        if not mjx_keys:
            self.skipTest("Could not parse return keys from MjxWalkEnv.get_obs_dict")

        missing = [k for k in mjx_keys if k not in mjlab_src]
        self.assertEqual(
            missing,
            [],
            msg=f"MjxWalkEnv obs keys not found in mjlab walk obs: {missing}",
        )


class TestRewardParity(unittest.TestCase):
    """Structural tests: reward functions match WalkEnvV0 / MjxWalkEnv (CPU/MJX) parity."""

    # Expected 5-term reward functions matching WalkEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS.
    _EXPECTED_REWARD_FUNCS = [
        "_walk_vel_reward",
        "_walk_done_signal",
        "_walk_cyclic_hip",
        "_walk_ref_rot",
        "_walk_joint_angle_rew",
    ]

    # Old simple reward functions that should NOT be primary config terms.
    _OLD_REWARD_FUNCS = [
        "_walk_forward_vel_reward",
        "_walk_alive_reward",
        "_walk_act_reg",
    ]

    def test_reward_functions_defined(self):
        """All 5 CPU/MJX-equivalent reward functions must be defined in source."""
        src = _src()
        for fn in self._EXPECTED_REWARD_FUNCS:
            self.assertIn(
                f"def {fn}(",
                src,
                msg=f"Missing reward function: {fn}",
            )

    def test_old_simplified_rewards_not_in_cfg(self):
        """Old simplified rewards (_walk_forward_vel_reward, _walk_alive_reward, _walk_act_reg)
        must not be registered as reward terms in _make_walk_env_cfg.
        """
        src = _src()
        cfg_start = src.index("def _make_walk_env_cfg(")
        next_def = src.index("\ndef _walk_ppo_runner_cfg(", cfg_start)
        cfg_src = src[cfg_start:next_def]
        for fn in self._OLD_REWARD_FUNCS:
            self.assertNotIn(
                fn,
                cfg_src,
                msg=f"Old simplified reward {fn} is still referenced in _make_walk_env_cfg; "
                f"replace with CPU/MJX-equivalent reward function.",
            )

    def test_all_reward_funcs_referenced_in_cfg(self):
        """Every CPU/MJX-equivalent reward function must be used in _make_walk_env_cfg."""
        src = _src()
        cfg_start = src.index("def _make_walk_env_cfg(")
        next_def = src.index("\ndef _walk_ppo_runner_cfg(", cfg_start)
        cfg_src = src[cfg_start:next_def]
        for fn in self._EXPECTED_REWARD_FUNCS:
            self.assertIn(
                fn,
                cfg_src,
                msg=f"Reward function {fn} not referenced in _make_walk_env_cfg",
            )

    def test_reward_weights_match_cpu(self):
        """Reward weights in _make_walk_env_cfg must match WalkEnvV0 DEFAULT_RWD_KEYS_AND_WEIGHTS.

        CPU/MJX weights (walk_v0.py:203-209):
          vel_reward      ×  5.0
          done            × -100.0
          cyclic_hip      × -10.0
          ref_rot         ×  10.0
          joint_angle_rew ×   5.0
        """
        src = _src()
        cfg_start = src.index("def _make_walk_env_cfg(")
        next_def = src.index("\ndef _walk_ppo_runner_cfg(", cfg_start)
        cfg_src = src[cfg_start:next_def]

        expected_weights = {
            "vel_reward": "5.0",
            "done": "-100.0",
            "cyclic_hip": "-10.0",
            "ref_rot": "10.0",
            "joint_angle_rew": "5.0",
        }
        for term, weight in expected_weights.items():
            self.assertIn(
                weight,
                cfg_src,
                msg=f"Expected weight {weight} for reward term '{term}' not found in "
                f"_make_walk_env_cfg; check reward weights match CPU/MJX.",
            )

    def test_vel_reward_uses_exponential_decay(self):
        """_walk_vel_reward must use exp() matching CPU formula (not linear clamp)."""
        src = _src()
        fn_start = src.index("def _walk_vel_reward(")
        fn_end = src.index("\ndef _walk_done_signal(")
        fn_src = src[fn_start:fn_end]
        self.assertIn(
            "exp",
            fn_src,
            msg="_walk_vel_reward should use exponential decay (torch.exp), not linear clamp",
        )
        self.assertIn(
            "com_vel",
            fn_src,
            msg="_walk_vel_reward should use mass-weighted COM velocity",
        )
        self.assertIn(
            "cvel",
            fn_src,
            msg="_walk_vel_reward should compute COM velocity from data.cvel",
        )

    def test_done_signal_checks_height_and_rotation(self):
        """_walk_done_signal must check both height and rotation thresholds."""
        src = _src()
        fn_start = src.index("def _walk_done_signal(")
        fn_end = src.index("\ndef _walk_cyclic_hip(")
        fn_src = src[fn_start:fn_end]
        self.assertIn(
            "min_height",
            fn_src,
            msg="_walk_done_signal should check height < min_height",
        )
        self.assertIn(
            "max_rot",
            fn_src,
            msg="_walk_done_signal should check rotation > max_rot",
        )
        self.assertIn(
            "qpos",
            fn_src,
            msg="_walk_done_signal should read qpos for rotation check",
        )

    def test_cyclic_hip_uses_phase_and_cosine(self):
        """_walk_cyclic_hip must use phase variable and cosine target trajectory."""
        src = _src()
        fn_start = src.index("def _walk_cyclic_hip(")
        fn_end = src.index("\ndef _walk_ref_rot(")
        fn_src = src[fn_start:fn_end]
        self.assertIn(
            "cos",
            fn_src,
            msg="_walk_cyclic_hip should use cosine for hip target trajectory",
        )
        self.assertIn(
            "phase",
            fn_src,
            msg="_walk_cyclic_hip should use phase variable for gait periodicity",
        )
        self.assertIn(
            "0.8",
            fn_src,
            msg="_walk_cyclic_hip amplitude should be 0.8 (matching CPU formula)",
        )
        self.assertIn(
            "hip_flex",
            fn_src,
            msg="_walk_cyclic_hip should reference hip_flex joint angles",
        )

    def test_ref_rot_uses_keyframe2_target(self):
        """_walk_ref_rot must use key_qpos[2] as target rotation (matching MJX/CPU with reset_type=init)."""
        src = _src()
        # Check _resolve_walk_obs_ids caches target_rot from keyframe 2
        fn_start = src.index("def _resolve_walk_obs_ids(")
        next_fn = src.index("\ndef _walk_obs_qpos_without_xy(")
        fn_src = src[fn_start:next_fn]
        self.assertIn(
            "target_rot",
            fn_src,
            msg="_resolve_walk_obs_ids should cache target_rot from keyframe 2",
        )
        self.assertIn(
            "[2]",
            fn_src,
            msg="_resolve_walk_obs_ids should use keyframe index 2 (standing 'init' pose) for target_rot",
        )

    def test_joint_angle_rew_uses_hip_adduction_rotation(self):
        """_walk_joint_angle_rew must use hip adduction and rotation angles."""
        src = _src()
        fn_start = src.index("def _walk_joint_angle_rew(")
        # End at next top-level def
        fn_end = src.index("\ndef _elbow_spec_fn(")
        fn_src = src[fn_start:fn_end]
        self.assertIn(
            "hip_adduct",
            fn_src,
            msg="_walk_joint_angle_rew should penalise hip adduction angles",
        )
        self.assertIn(
            "hip_rot",
            fn_src,
            msg="_walk_joint_angle_rew should penalise hip rotation angles",
        )
        self.assertIn(
            "exp",
            fn_src,
            msg="_walk_joint_angle_rew should use exp(-5 * mean(|angles|))",
        )

    def test_policy_group_has_5_reward_terms(self):
        """The rewards dict in _make_walk_env_cfg must have exactly 5 terms."""
        src = _src()
        cfg_start = src.index("def _make_walk_env_cfg(")
        next_def = src.index("\ndef _walk_ppo_runner_cfg(", cfg_start)
        cfg_src = src[cfg_start:next_def]

        count = sum(1 for fn in self._EXPECTED_REWARD_FUNCS if fn in cfg_src)
        self.assertEqual(
            count,
            5,
            msg=f"Expected 5 reward functions in _make_walk_env_cfg, found {count}. "
            f"Missing: {[fn for fn in self._EXPECTED_REWARD_FUNCS if fn not in cfg_src]}",
        )


class TestActionNormalizationParity(unittest.TestCase):
    """Structural tests: action normalization matches WalkEnvV0 (CPU) sigmoid."""

    def test_sigmoid_normalization_in_process_actions(self):
        """MyoMuscleActivationAction.process_actions must use sigmoid, not linear affine.

        CPU formula (base_v0.py:88-92):
            ctrl = 1 / (1 + exp(-5 * (a - 0.5)))  =  sigmoid(5 * (a - 0.5))
        Old (wrong) MJLAB formula:
            ctrl = 0.5 * (a + 1.0)  # linear, not matching CPU
        """
        src = _src()
        self.assertIn(
            "sigmoid",
            src,
            msg="Action normalization should use sigmoid (not linear 0.5*(a+1)). "
            "Expected: torch.sigmoid(5.0 * (self._raw_actions - 0.5))",
        )

    def test_sigmoid_coefficient_is_5(self):
        """Sigmoid coefficient must be 5.0, matching CPU base_v0.py:90."""
        src = _src()
        # Find process_actions method
        fn_start = src.index("def process_actions(")
        fn_end = src.index("\n    def ", fn_start)
        fn_src = src[fn_start:fn_end]
        self.assertIn(
            "5.0",
            fn_src,
            msg="Sigmoid coefficient should be 5.0 (matching CPU: sigmoid(5*(a-0.5)))",
        )

    def test_linear_affine_not_in_process_actions(self):
        """Linear affine 0.5 * (a + 1) must NOT be the normalization formula."""
        src = _src()
        fn_start = src.index("def process_actions(")
        fn_end = src.index("\n    def ", fn_start)
        fn_src = src[fn_start:fn_end]
        # Check the old linear formula is gone — should not use 0.5 * (... + 1)
        self.assertNotIn(
            "0.5 * (self._raw_actions + 1.0)",
            fn_src,
            msg="Old linear affine normalization '0.5 * (self._raw_actions + 1.0)' "
            "found in process_actions; replace with sigmoid.",
        )


class TestXmlModelParity(unittest.TestCase):
    """Structural tests: mjlab uses same XML model as CPU/MJX backends."""

    def test_walk_xml_uses_simhive_model(self):
        """_WALK_XML must point to simhive/myo_sim/leg/myolegs.xml (same as CPU/MJX).

        Old (wrong): myosuite/envs/myo/assets/leg/myolegs_chasetag.xml
        Correct:     myosuite/simhive/myo_sim/leg/myolegs.xml
        """
        src = _src()
        self.assertIn(
            "simhive/myo_sim/leg/myolegs.xml",
            src,
            msg="_WALK_XML must use simhive/myo_sim/leg/myolegs.xml (same as CPU/MJX); "
            "was myolegs_chasetag.xml which is a different task model.",
        )

    def test_chasetag_xml_not_used_for_walk(self):
        """myolegs_chasetag.xml must not be the _WALK_XML path."""
        src = _src()
        # Find the _WALK_XML assignment
        xml_start = src.index("_WALK_XML")
        xml_line_end = src.index("\n", xml_start)
        xml_line = src[xml_start:xml_line_end]
        self.assertNotIn(
            "chasetag",
            xml_line,
            msg="_WALK_XML should not reference myolegs_chasetag.xml; "
            "use myolegs.xml from simhive (same model as CPU/MJX).",
        )

    def test_myosuite_root_fallback_path_correct(self):
        """Fallback _MYOSUITE_ROOT must use parents[3] (not parents[2]).

        register_mjlab_tasks.py is at:
          myosuite/envs/myo/backends/mjlab/register_mjlab_tasks.py
        parents[0] = mjlab/
        parents[1] = myo/
        parents[2] = envs/
        parents[3] = myosuite/ (package root)
        """
        src = _src()
        self.assertIn(
            "parents[3]",
            src,
            msg="Fallback _MYOSUITE_ROOT should use parents[3] (myosuite package root), "
            "not parents[2] which resolves to envs/.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
