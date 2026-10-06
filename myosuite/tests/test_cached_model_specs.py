# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""gym.make reuses cached model specs: kept from the second build, a private copy per env."""

from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np
import pytest

import myosuite  # noqa: F401  (registers the envs)
from myosuite.core import model_recipes
from myosuite.core.muscle_conditions import _peak_force
from myosuite.integrations.musclemimic import fullbody_model
from myosuite.utils import gym
from myosuite import make_env

pytestmark = pytest.mark.tier1

TT_P0 = "myoChallengeTableTennisP0-v0"
CHASETAG_FB = "myoChallengeChaseTagFBP2-v0"


def _count_calls(monkeypatch: pytest.MonkeyPatch, module: object, name: str) -> list:
    """Wrap ``module.name`` to record each call; returns the call log."""
    calls: list = []
    fn: Callable = getattr(module, name)

    def _counted(*args, **kwargs):
        calls.append(args)
        return fn(*args, **kwargs)

    monkeypatch.setattr(module, name, _counted)
    return calls


def _arrays(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    """Every numpy array attribute of *model*."""
    names = (name for name in dir(model) if not name.startswith("_"))
    values = {name: getattr(model, name) for name in names}
    return {k: v for k, v in values.items() if isinstance(v, np.ndarray)}


def _assert_same_model(a: mujoco.MjModel, b: mujoco.MjModel) -> None:
    """Every array attribute and every ``opt``/``stat`` field is identical."""
    arrays_a, arrays_b = _arrays(a), _arrays(b)
    assert arrays_a.keys() == arrays_b.keys()
    for name, value in arrays_a.items():
        np.testing.assert_array_equal(value, arrays_b[name], err_msg=name)
    for struct in ("opt", "stat"):
        for name in dir(getattr(a, struct)):
            if not name.startswith("_"):
                np.testing.assert_array_equal(
                    getattr(getattr(a, struct), name),
                    getattr(getattr(b, struct), name),
                    err_msg=f"{struct}.{name}",
                )


def _make_model(env_id: str) -> tuple[gym.Env, mujoco.MjModel]:
    env = make_env(env_id)
    return env, env.unwrapped.model


def test_tabletennis_builds_the_recipe_twice_for_three_makes(monkeypatch):
    """The TableTennis body is kept from its second build; all makes compile equal models."""
    from myosuite.core.model_builder import clear_spec_caches

    # Start without kept specs or build counts. (Not a chdir to a new cache key: the
    # TableTennis furniture mesh paths are cwd-relative and fail across drives.)
    clear_spec_caches()
    calls = _count_calls(monkeypatch, model_recipes, "_tabletennis_body_spec")
    envs, models = zip(*(_make_model(TT_P0) for _ in range(3)))
    assert len(calls) == 2
    assert models[2] is not models[1]
    for model in models[1:]:
        _assert_same_model(models[0], model)
    for env in envs:
        env.close()


def test_a_single_make_keeps_no_recipe_spec() -> None:
    """A process that makes TableTennis once (a vector-env worker) caches nothing."""
    from myosuite.core.model_builder import _recipe_spec, clear_spec_caches

    clear_spec_caches()  # as in a fresh process
    env, _ = _make_model(TT_P0)
    assert _recipe_spec.cache_info().currsize == 0
    env.close()


def test_chasetag_fb_loads_the_fullbody_once_for_two_makes(monkeypatch):
    """Two makes load and edit the MuscleMimic full body once."""
    from myosuite.core.model_builder import clear_spec_caches

    clear_spec_caches()
    calls = _count_calls(
        monkeypatch, fullbody_model, "_load_spec_with_floor_only_upstream_scene"
    )
    env_a, model_a = _make_model(CHASETAG_FB)
    env_b, model_b = _make_model(CHASETAG_FB)
    assert len(calls) == 1
    _assert_same_model(model_a, model_b)
    env_a.close()
    env_b.close()


@pytest.mark.parametrize("env_id", [TT_P0, CHASETAG_FB])
def test_cached_makes_share_no_model_or_spec_state(env_id: str) -> None:
    """Edits to one env's model and spec never reach a later make of the id."""
    from myosuite.core.model_builder import clear_spec_caches

    warm = [_make_model(env_id)[0] for _ in range(2)]  # the spec is kept by now
    env_a, model_a = _make_model(env_id)
    model_a.geom_size[:] *= 2.0
    model_a.body_mass[:] *= 3.0
    spec_a = env_a.unwrapped._mj_spec
    spec_a.worldbody.add_body(name="edit_of_a")
    spec_a.geoms[0].size = [9.0, 9.0, 9.0]
    env_b, model_b = _make_model(env_id)
    clear_spec_caches()
    env_fresh, model_fresh = _make_model(env_id)

    _assert_same_model(model_b, model_fresh)
    assert not np.shares_memory(model_b.geom_size, model_a.geom_size)
    spec_b = env_b.unwrapped._mj_spec
    assert spec_b is not spec_a
    assert all(body.name != "edit_of_a" for body in spec_b.bodies)
    np.testing.assert_array_equal(
        spec_b.geoms[0].size, env_fresh.unwrapped._mj_spec.geoms[0].size
    )
    for env in (*warm, env_a, env_b, env_fresh):
        env.close()


def test_muscle_condition_variant_does_not_reach_the_cached_spec() -> None:
    """The sarcopenia variant edits its own model copy, not the shared recipe spec."""
    env_base, model_base = _make_model(TT_P0)
    _make_model(TT_P0)[0].close()  # the recipe spec is kept by now
    env_sarc, model_sarc = _make_model("myoSarcChallengeTableTennisP0-v0")
    np.testing.assert_allclose(
        model_sarc.actuator_gainprm[:, 2], 0.5 * _peak_force(model_base)
    )
    env_again, model_again = _make_model(TT_P0)
    _assert_same_model(model_base, model_again)
    for env in (env_sarc, env_base, env_again):
        env.close()


def test_distinct_cache_keys_build_distinct_specs() -> None:
    """Different recipes and full-body configs never share a cached spec."""
    from myosuite.core.model_builder import build_from_recipe

    pose, _ = build_from_recipe("hand_pose")
    sar, _ = build_from_recipe("hand_sar")
    assert (pose.nbody, pose.nsite) != (sar.nbody, sar.nsite)

    config = fullbody_model.default_mimic_fullbody_config()
    without_fingers = fullbody_model.build_mimic_fullbody_spec(config)[0].compile()
    config.disable_fingers = False
    with_fingers = fullbody_model.build_mimic_fullbody_spec(config)[0].compile()
    assert with_fingers.nu > without_fingers.nu
    assert with_fingers.njnt > without_fingers.njnt


def test_native_fallback_specs_are_kept_per_disable_fingers() -> None:
    from ml_collections import config_dict

    from myosuite.core.model_builder import clear_spec_caches
    from myosuite.integrations.musclemimic import bimanual_model

    clear_spec_caches()
    for module, build, native in (
        (fullbody_model, "build_native_mimic_fullbody_spec", "_native_fullbody_spec"),
        (bimanual_model, "build_native_mimic_bimanual_spec", "_native_bimanual_spec"),
    ):
        cfg = config_dict.create(disable_fingers=True)
        first, second, third = (getattr(module, build)(cfg) for _ in range(3))
        assert first is not second and second is not third
        assert getattr(module, native).cache_info().currsize == 1
        n_actuators = first.compile().nu
        assert second.compile().nu == third.compile().nu == n_actuators
        with_fingers = getattr(module, build)(config_dict.create(disable_fingers=False))
        assert with_fingers.compile().nu > n_actuators


def test_bimanual_spec_cache_follows_the_scene_variable(monkeypatch) -> None:
    pytest.importorskip("musclemimic_models")
    from ml_collections import config_dict

    from myosuite.integrations.musclemimic import bimanual_model

    bimanual_model._mimic_bimanual_spec.cache_clear()
    cfg = config_dict.create(disable_fingers=True, model_path=None)
    monkeypatch.delenv("MYOSUITE_MIMIC_STRICT_UPSTREAM_SCENE", raising=False)
    bimanual_model.build_mimic_bimanual_spec(cfg)
    monkeypatch.setenv("MYOSUITE_MIMIC_STRICT_UPSTREAM_SCENE", "1")
    bimanual_model.build_mimic_bimanual_spec(cfg)
    assert bimanual_model._mimic_bimanual_spec.cache_info().misses == 2


def test_mjlab_tabletennis_full_spec_is_built_once_for_three_calls(monkeypatch) -> None:
    pytest.importorskip("mjlab")
    from myosuite.core.model_builder import clear_spec_caches
    from myosuite.envs.myo.backends.mjlab import register_mjlab_tabletennis as tt

    clear_spec_caches()
    calls = _count_calls(monkeypatch, tt, "_add_tabletennis_furniture")
    specs = [tt._table_tennis_full_spec() for _ in range(3)]
    assert len(calls) == 2  # kept from the second build
    assert len({id(s) for s in specs}) == 3  # each caller gets its own spec
