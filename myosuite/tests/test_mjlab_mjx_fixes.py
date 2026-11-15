# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests for mjlab multi-env origins, horizons, host syncs, and mjx fixes."""

from __future__ import annotations

import ast
import importlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.tier1

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MJLAB_DIR = _REPO_ROOT / "myosuite" / "envs" / "myo" / "backends" / "mjlab"
_MJX_DIR = _REPO_ROOT / "myosuite" / "envs" / "myo" / "backends" / "mjx"


def test_episode_length_formula_avoids_ieee754_ceil_overshoot() -> None:
    """(N - 0.5) * dt keeps math.ceil(episode_length_s / dt) == N across horizons."""
    assert any(math.ceil((n * 0.02) / 0.02) == n + 1 for n in range(1, 200))
    for dt in (0.01, 0.02, 0.012):
        for n_steps in range(1, 1001):
            episode_len = (n_steps - 0.5) * dt
            assert math.ceil(episode_len / dt) == n_steps


def test_mjlab_modules_import_without_torch_or_mjlab() -> None:
    """mjlab_env_base and mimic_mjlab_env must remain importable on CPU-only installs."""
    env_base = importlib.import_module(
        "myosuite.envs.myo.backends.mjlab.mjlab_env_base"
    )
    mimic = importlib.import_module("myosuite.envs.myo.backends.mjlab.mimic_mjlab_env")
    assert hasattr(env_base, "normalize_mjlab_env_ids")
    assert hasattr(mimic, "_mimic_cache_key")


def test_mimic_cache_key_isolates_instances_sharing_cfg() -> None:
    """Two envs sharing one cfg must not collide, while wrappers share unwrapped."""
    mimic = importlib.import_module("myosuite.envs.myo.backends.mjlab.mimic_mjlab_env")
    shared_cfg = SimpleNamespace(decimation=5)
    env_a = SimpleNamespace(cfg=shared_cfg)
    env_b = SimpleNamespace(cfg=shared_cfg)
    wrapped_a = SimpleNamespace(cfg=shared_cfg, unwrapped=env_a)

    key_a = mimic._mimic_cache_key(env_a, "robot", "fullbody")
    key_b = mimic._mimic_cache_key(env_b, "robot", "fullbody")
    key_wrapped_a = mimic._mimic_cache_key(wrapped_a, "robot", "fullbody")

    assert key_a != key_b
    assert key_a == key_wrapped_a


def test_no_step_rate_any_host_sync_in_clip_trajectory_or_mimic_sync() -> None:
    """clip_trajectory_source and _sync_mimic_mjlab_targets must not call .any()."""
    clip_src = (_MJLAB_DIR / "clip_trajectory_source.py").read_text(encoding="utf-8")
    assert ".any()" not in clip_src

    mimic_tree = ast.parse(
        (_MJLAB_DIR / "mimic_mjlab_env.py").read_text(encoding="utf-8")
    )
    sync_fn = next(
        node
        for node in mimic_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_sync_mimic_mjlab_targets"
    )
    any_calls = [
        node
        for node in ast.walk(sync_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "any"
    ]
    assert not any_calls


def test_all_four_mjlab_builders_use_episode_length_helper() -> None:
    """mjlab_task_builder, mimic_mjlab_env (both builders), and tabletennis use helper."""
    for rel in (
        "mjlab_task_builder.py",
        "mimic_mjlab_env.py",
        "register_mjlab_tabletennis.py",
    ):
        text = (_MJLAB_DIR / rel).read_text(encoding="utf-8")
        assert "episode_length_s(" in text, f"{rel} missing episode_length_s helper"


def test_mjx_lazy_logging_and_modular_spec_assignment() -> None:
    """train_jax_ppo and mjx/utils avoid top-level tensorboardX/wandb imports."""
    train_tree = ast.parse((_MJX_DIR / "train_jax_ppo.py").read_text(encoding="utf-8"))
    top_imports = [
        alias.name
        for node in train_tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    ] + [
        node.module or ""
        for node in train_tree.body
        if isinstance(node, ast.ImportFrom)
    ]
    assert "tensorboardX" not in top_imports
    assert "wandb" not in top_imports

    utils_tree = ast.parse((_MJX_DIR / "utils.py").read_text(encoding="utf-8"))
    utils_imports = [
        alias.name
        for node in utils_tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert "wandb" not in utils_imports

    modular_src = (_MJX_DIR / "mjx_modular_env.py").read_text(encoding="utf-8")
    assert "self._mj_spec = spec" in modular_src


def test_tracked_site_pos_subtracts_env_origins_for_floating_base() -> None:
    """Floating-base mimic site reads subtract env_origins; fixed-base do not."""
    torch = pytest.importorskip("torch")
    mimic = importlib.import_module("myosuite.envs.myo.backends.mjlab.mimic_mjlab_env")

    origins = torch.tensor([[5.0, -3.0, 0.5], [-2.0, 4.0, 0.0]], dtype=torch.float32)
    world_sites = torch.tensor(
        [
            [[6.0, -2.0, 1.5], [5.5, -3.0, 1.0]],
            [[-1.0, 5.0, 1.0], [-2.0, 4.5, 0.5]],
        ],
        dtype=torch.float32,
    )
    cache: dict[str, object] = {"site_ids": np.array([0, 1], dtype=np.int32)}

    for fixed_base in (False, True):
        entity = SimpleNamespace(
            is_fixed_base=fixed_base,
            data=SimpleNamespace(site_pos_w=world_sites),
        )
        scene = type("Scene", (dict,), {})(robot=entity)
        scene.env_origins = origins
        env = SimpleNamespace(scene=scene)
        out = mimic._tracked_site_pos(env, "robot", cache)
        expected = world_sites if fixed_base else (world_sites - origins[:, None, :])
        torch.testing.assert_close(out, expected)


def test_entity_accessor_slices_entity_muscle_activations() -> None:
    """MjlabEntityAccessor.muscle_act returns only the entity's stateful activations."""
    torch = pytest.importorskip("torch")
    env_base = importlib.import_module(
        "myosuite.envs.myo.backends.mjlab.mjlab_env_base"
    )

    all_act = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    mj_model = SimpleNamespace(
        na=4,
        actuator_actadr=np.array([0, -1, 1, 2, 3], dtype=np.int32),
        actuator_actnum=np.array([1, 0, 1, 1, 1], dtype=np.int32),
    )
    entity = SimpleNamespace(
        is_fixed_base=True,
        data=SimpleNamespace(data=SimpleNamespace(act=all_act)),
        indexing=SimpleNamespace(ctrl_ids=torch.tensor([0, 1, 3], dtype=torch.long)),
    )
    env = SimpleNamespace(
        device=all_act.device,
        scene={"arm": entity},
        sim=SimpleNamespace(mj_model=mj_model),
    )
    acc = env_base.MjlabEntityAccessor(env, "arm")
    torch.testing.assert_close(acc.muscle_act(), all_act[:, [0, 2]])

    from myosuite.tests.support.host_sync import HostSyncCounter

    with HostSyncCounter(package_only=True) as syncs:
        fresh_acc = env_base.MjlabEntityAccessor(env, "arm")
        torch.testing.assert_close(fresh_acc.muscle_act(), all_act[:, [0, 2]])
    assert syncs.total == 0, syncs.report()
