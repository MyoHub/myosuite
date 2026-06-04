"""Static invariant checks for the mjlab training path.

These tests parse ``run_benchmark.py`` with the stdlib ``ast`` module to
assert that ``_run_mjlab_training``:

  - uses RSL-RL ``OnPolicyRunner`` (GPU-native on-policy PPO)
  - uses ``SARTorchTransform`` for the SAR inverse transform on GPU
  - does NOT use ``stable_baselines3`` / SB3 wrappers (CPU-side RL library)

**Zero optional dependencies required** — the tests read source text directly
via ``pathlib`` and ``ast`` (both stdlib).  No numpy, torch, rsl_rl, mjlab,
or gymnasium installation is needed.  Tests complete in under one second.

Run::

    pytest benchmarks/sar_backends/test_mjlab_training_invariants.py -v
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_BENCHMARK_FILE = pathlib.Path(__file__).parent / "run_benchmark.py"
_WALK_CFG_FILE = (
    pathlib.Path(__file__).parents[2]
    / "myosuite/envs/myo/backends/mjlab/register_mjlab_tasks.py"
)


def _training_src() -> str:
    """Extract the source of ``_run_mjlab_training`` without importing the module.

    Uses ``ast.parse`` to locate the function by name, then slices the raw
    source lines.  This avoids importing ``run_benchmark.py``, which would
    require numpy, gymnasium, and other heavy optional dependencies.
    """
    src = _BENCHMARK_FILE.read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_run_mjlab_training":
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise RuntimeError(
        f"_run_mjlab_training not found in {_BENCHMARK_FILE}. "
        "Has the function been renamed or removed?"
    )


def _fn_node() -> ast.FunctionDef:
    """Return the AST node for ``_run_mjlab_training``."""
    src = _BENCHMARK_FILE.read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_run_mjlab_training":
            return node
    raise RuntimeError(f"_run_mjlab_training not found in {_BENCHMARK_FILE}")


def _has_code_ref(symbol: str) -> bool:
    """Return True if ``symbol`` appears as a live code reference inside
    ``_run_mjlab_training`` — i.e. as an import name, ``ast.Name``, or
    ``ast.Attribute``.  String constants (docstrings, comments) are excluded
    because ``ast.walk`` never yields them as Name/Attribute/Import nodes.
    """
    fn = _fn_node()
    for node in ast.walk(fn):
        if isinstance(node, ast.ImportFrom):
            if node.module and symbol in node.module:
                return True
            if any(symbol in alias.name for alias in node.names):
                return True
        elif isinstance(node, ast.Import):
            if any(symbol in alias.name for alias in node.names):
                return True
        elif isinstance(node, ast.Name) and node.id == symbol:
            return True
        elif isinstance(node, ast.Attribute) and node.attr == symbol:
            return True
    return False


# ---------------------------------------------------------------------------
# Positive assertions: things that MUST be present
# ---------------------------------------------------------------------------


class TestMjlabTrainingMustUseRslRl:
    """The mjlab training path must be driven by RSL-RL PPO."""

    def test_imports_rslrl(self):
        """``rsl_rl`` must be imported inside the function."""
        assert "rsl_rl" in _training_src(), (
            "_run_mjlab_training must import from rsl_rl. "
            "The mjlab backend is trained via RSL-RL PPO (GPU-native), not SB3."
        )

    def test_uses_on_policy_runner(self):
        """``OnPolicyRunner`` (RSL-RL's PPO runner) must be referenced."""
        assert "OnPolicyRunner" in _training_src(), (
            "_run_mjlab_training must use rsl_rl.runners.OnPolicyRunner. "
            "SB3 SAC is for the CPU/MJX backends only."
        )

    def test_calls_runner_learn(self):
        """``runner.learn(`` must appear — that's what drives the PPO loop."""
        assert "runner.learn(" in _training_src(), (
            "_run_mjlab_training must call runner.learn() to drive training. "
            "If this is missing the training loop was not wired up."
        )


class TestMjlabTrainingMustUseGpuSar:
    """The SAR inverse transform must run on GPU (torch matmul), not sklearn on CPU."""

    def test_uses_sar_torch_transform(self):
        """``SARTorchTransform`` must be imported and used."""
        assert "SARTorchTransform" in _training_src(), (
            "_run_mjlab_training must use SARTorchTransform (GPU-resident SAR). "
            "The sklearn-based synergy_inverse_transform runs on CPU and must "
            "not appear in the mjlab training hot path."
        )

    def test_has_sar_env_wrapper(self):
        """``_SAREnvWrapper`` inner class must exist to apply SAR before env.step()."""
        assert "_SAREnvWrapper" in _training_src(), (
            "_run_mjlab_training must define _SAREnvWrapper, which intercepts "
            "step() to apply SARTorchTransform on GPU before calling the mjlab env."
        )


# ---------------------------------------------------------------------------
# Negative assertions: things that must NOT be present
# ---------------------------------------------------------------------------


class TestMjlabTrainingMustNotUseSb3:
    """SB3 is a CPU-side RL library and must not appear in the mjlab training path."""

    def test_no_stable_baselines3_import(self):
        assert "stable_baselines3" not in _training_src(), (
            "_run_mjlab_training must NOT import stable_baselines3. "
            "SB3 SAC is used for the cpu/mjx_xla/mjx_warp backends. "
            "The mjlab backend uses RSL-RL PPO."
        )

    def test_no_sac(self):
        # Guard against someone importing SAC directly or aliased
        src = _training_src()
        assert "from stable_baselines3 import SAC" not in src
        assert "sb3.SAC" not in src

    @pytest.mark.parametrize(
        "wrapper",
        [
            "ActionWrapper",
            "DummyVecEnv",
            "VecNormalize",
        ],
    )
    def test_no_gymnasium_sb3_wrappers(self, wrapper):
        """Gymnasium/SB3 vectorisation wrappers must not appear."""
        assert wrapper not in _training_src(), (
            f"_run_mjlab_training must NOT use gymnasium/SB3 wrapper '{wrapper}'. "
            "These wrappers are part of the SB3/gymnasium CPU training stack. "
            "The mjlab backend is RSL-RL-native and uses _SAREnvWrapper instead."
        )


# ---------------------------------------------------------------------------
# Walk task config invariants — reward terms must be wired up
# ---------------------------------------------------------------------------


def _walk_cfg_src() -> str:
    """Extract the source of ``_make_walk_env_cfg`` from register_mjlab_tasks.py."""
    src = _WALK_CFG_FILE.read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_make_walk_env_cfg":
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise RuntimeError(
        f"_make_walk_env_cfg not found in {_WALK_CFG_FILE}. "
        "Has the function been renamed?"
    )


class TestWalkCfgMustHaveRewards:
    """``_make_walk_env_cfg`` must define reward terms.

    Without a ``rewards`` dict in the ``ManagerBasedRlEnvCfg``, mjlab returns
    a constant 0.0 reward every step regardless of what the agent does —
    making RL training impossible and causing user-visible confusion when
    observations change but rewards never do.
    """

    def test_rewards_key_passed_to_cfg(self):
        """``ManagerBasedRlEnvCfg(...)`` call must include ``rewards=``."""
        src = _walk_cfg_src()
        assert "rewards=" in src, (
            "_make_walk_env_cfg must pass a 'rewards' dict to ManagerBasedRlEnvCfg. "
            "Without reward terms, mjlab returns 0.0 every step."
        )

    def test_reward_term_cfg_used(self):
        """``RewardTermCfg`` must appear in the function."""
        src = _walk_cfg_src()
        assert "RewardTermCfg" in src, (
            "_make_walk_env_cfg must use RewardTermCfg to define reward terms. "
            "Found no reference to RewardTermCfg."
        )

    def test_forward_vel_reward_present(self):
        """A forward velocity reward function must be wired into the config."""
        src = _walk_cfg_src()
        assert "_walk_forward_vel_reward" in src, (
            "_make_walk_env_cfg must include a forward velocity reward. "
            "Without it the agent has no incentive to walk forward."
        )

    def test_alive_reward_present(self):
        """An alive (height) reward must be wired into the config."""
        src = _walk_cfg_src()
        assert "_walk_alive_reward" in src, (
            "_make_walk_env_cfg must include an alive/height reward. "
            "Without it the agent has no incentive to stay upright."
        )

    def test_walk_cfg_params_used(self):
        """``WalkCfg`` reward parameters must be referenced (not hard-coded magic numbers)."""
        src = _walk_cfg_src()
        assert "WalkCfg" in src, (
            "_make_walk_env_cfg should instantiate WalkCfg to source reward parameters "
            "(target_vel, alive_bonus, act_reg_weight, fall_height_threshold). "
            "This keeps reward hyper-parameters in one place."
        )


class TestWalkRolloutMustCallReset:
    """``_run_mjlab_rollout`` must call ``reset()`` before ``step()``.

    Without ``reset()``, the mjlab physics state is uninitialized and every
    ``step()`` returns the same observation and reward regardless of action.
    """

    def _rollout_src(self) -> str:
        src = _BENCHMARK_FILE
        full = src.read_text()
        tree = ast.parse(full)
        lines = full.splitlines()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "_run_mjlab_rollout":
                return "\n".join(lines[node.lineno - 1 : node.end_lineno])
        raise RuntimeError("_run_mjlab_rollout not found")

    def test_reset_called_before_step(self):
        src = self._rollout_src()
        assert "reset(" in src, (
            "_run_mjlab_rollout must call env.reset() before the first step(). "
            "Without reset(), physics state is uninitialized and observations "
            "never change."
        )
