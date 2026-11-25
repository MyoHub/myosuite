# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Mixtures of policy snapshots (``myosuite.utils.policy_mixture``) and their use in the scripts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from myosuite.utils.checkpoint_utils import find_vec_normalize
from myosuite.utils.policy_mixture import PolicyMixture, select_snapshots

pytestmark = pytest.mark.tier1

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("pattern", ["model_{}.pt", "model_{}_steps.zip"])
def test_snapshots_are_spread_over_the_last_part_of_the_run(
    tmp_path: Path, pattern: str
) -> None:
    for step in range(100, 1100, 100):
        (tmp_path / pattern.format(step)).write_bytes(b"")
    for other in (
        "model.zip",
        "model_vecnormalize_900_steps.pkl",
        "events.out",
        "best_2000.pt",
    ):
        (tmp_path / other).write_bytes(b"")
    names = [p.name for p in select_snapshots(tmp_path, 3, last=0.5)]
    assert names == [pattern.format(s) for s in (600, 800, 1000)]
    names = [p.name for p in select_snapshots(tmp_path, 2, last=0.5, until=800)]
    assert names == [pattern.format(s) for s in (500, 800)]
    assert len(select_snapshots(tmp_path, 10, last=1.0)) == 10
    with pytest.raises(ValueError, match="5 snapshots"):
        select_snapshots(tmp_path, 6, last=0.5)
    with pytest.raises(ValueError, match="last"):
        select_snapshots(tmp_path, 2, last=0.0)


def test_a_dotted_sb3_name_keeps_its_step() -> None:
    from myosuite.utils.policy_mixture import snapshot_step

    assert snapshot_step(Path("run.v1_4000_steps.zip")) == 4000
    assert snapshot_step(Path("model_400.pt")) == 400
    assert snapshot_step(Path("best_400.pt")) is None


def test_a_snapshot_finds_its_own_normaliser(tmp_path: Path) -> None:
    snapshot = tmp_path / "model_400_steps.zip"
    for name in ("vecnormalize.pkl", "model_vecnormalize_400_steps.pkl"):
        (tmp_path / name).write_bytes(b"")
        assert find_vec_normalize(snapshot) == tmp_path / name


def test_each_env_is_run_by_its_own_policy() -> None:
    policies = [lambda obs, k=k: obs * 0 + k for k in range(3)]
    mixture = PolicyMixture(policies, n_envs=600, seed=1)
    obs = np.ones((600, 2))
    actions = mixture.apply(lambda p, o: p(o), obs)
    np.testing.assert_array_equal(actions[:, 0], mixture.index)
    assert np.bincount(mixture.index, minlength=3).min() > 150  # uniform draws
    before = mixture.index.copy()
    mixture.redraw([0, 1, 2])
    mixture.redraw([])  # no episode ended
    np.testing.assert_array_equal(mixture.index[3:], before[3:])
    single = PolicyMixture(policies, seed=2)
    assert single(np.ones(2))[0] == single.index[0]
    with pytest.raises(ValueError, match="at least one"):
        PolicyMixture([])
    with pytest.raises(ValueError, match="apply"):
        PolicyMixture(policies, n_envs=2)(np.ones((2, 2)))


def test_the_mixture_runs_on_torch() -> None:
    torch = pytest.importorskip("torch")
    policies = [lambda obs, k=k: obs[:, :1] * 0 + k for k in range(2)]
    mixture = PolicyMixture(policies, n_envs=8, seed=3)
    actions = mixture.apply(lambda p, o: p(o), torch.ones(8, 3))
    assert isinstance(actions, torch.Tensor) and actions.shape == (8, 1)
    np.testing.assert_array_equal(actions[:, 0].numpy(), mixture.index)


@pytest.mark.tier2
def test_the_evaluation_script_mixes_the_snapshots_of_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("tyro")
    from myosuite import make_env
    from myosuite.utils import rslrl_policy

    env_id = "myoElbowPose1D6MRandom-v0"
    env = make_env(env_id)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    env.close()
    for k, step in enumerate((100, 200, 300, 400)):
        actor = {
            "mlp.0.weight": torch.zeros(4, obs_dim),
            "mlp.0.bias": torch.zeros(4),
            "mlp.2.weight": torch.zeros(act_dim, 4),
            "mlp.2.bias": torch.full((act_dim,), float(k)),
        }
        torch.save({"actor_state_dict": actor}, tmp_path / f"model_{step}.pt")
    loaded: list[str] = []
    load = rslrl_policy.load_rslrl_policy

    def recording(path: Path, *args: object, **kwargs: object) -> object:
        loaded.append(Path(path).name)
        return load(path, *args, **kwargs)

    monkeypatch.setattr(rslrl_policy, "load_rslrl_policy", recording)
    script = _load("eval_mjlab_policy")
    cfg = script.EvalConfig(env_id, checkpoint=tmp_path, mixture=2, episodes=4)
    script.evaluate_cpu(cfg, script._resolve_checkpoint(env_id, tmp_path))
    assert loaded == ["model_300.pt", "model_400.pt"]
    assert "episodes: 4" in capsys.readouterr().out


def test_one_policy_for_all_envs_gets_the_observations_as_they_are() -> None:
    """The evaluation of a single checkpoint: no gather and scatter of the batch."""
    seen = []
    mixture = PolicyMixture(["only"], n_envs=3)
    obs = np.zeros((3, 2))
    out = mixture.apply(lambda policy, o: seen.append(o) or o + 1.0, obs)
    assert seen[0] is obs and out.shape == (3, 2)
