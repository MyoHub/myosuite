"""DynSyn muscle synergies: grouping, action mapping, CPU wrapper and mjlab action term."""

import re
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from myosuite.core.dynsyn import (
    DynSynGroups,
    expand_actions,
    k_medoids,
    length_similarity,
    synergy_groups,
    weight_amplitude,
)

GROUPS = [[0, 2], [1], [3, 4, 5]]


def test_expand_actions_numpy_matches_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    a = rng.normal(size=(7, 6))
    out_np = expand_actions(a, GROUPS, 0.05, np)
    out_t = expand_actions(torch.tensor(a), GROUPS, 0.05, torch).numpy()
    np.testing.assert_allclose(out_np, out_t, rtol=1e-12)


def test_expand_actions_groups_and_bounds():
    a = np.array([[0.5, -1.0, 2.0, 30.0, -30.0, 0.0]])  # 3 group actions, 3 corrections
    plain = expand_actions(a, GROUPS, 0.0, np)
    np.testing.assert_allclose(plain, [[0.5, -1.0, 0.5, 2.0, 2.0, 2.0]])
    corrected = expand_actions(a, GROUPS, 0.05, np)
    # representatives (0, 1, 3) keep their group action; the others are scaled within +-5 %
    np.testing.assert_allclose(corrected[0, [0, 1, 3]], [0.5, -1.0, 2.0])
    np.testing.assert_allclose(corrected[0, 2], 0.5 * 1.05)
    np.testing.assert_allclose(corrected[0, [4, 5]], [2.0 * 0.95, 2.0])


def test_max_amplitude_above_the_default_is_reached():
    a = np.array([[1.0, 1.0, 1.0, 30.0, 30.0, 30.0]])
    out = expand_actions(a, GROUPS, 0.3, np, kappa=0.3)
    np.testing.assert_allclose(out[0, [2, 4, 5]], 1.3)


def test_correction_scales_the_action_not_the_excitation():
    # A saturated positive correction: lower on a negative action, none on 0, higher on a positive one.
    a = np.array([[-0.5, 0.0, 0.5, 30.0, 30.0, 30.0]])
    out = expand_actions(a, [[0, 3], [1, 4], [2, 5]], 0.1, np)
    np.testing.assert_allclose(out[0, 3:], [-0.55, 0.0, 0.55])


def test_weight_amplitude_schedule():
    assert weight_amplitude(0, 100, 200) == 0.0
    assert weight_amplitude(100, 100, 200) == 0.0
    assert weight_amplitude(200, 100, 200) == pytest.approx(0.05)
    assert weight_amplitude(1e9, 100, 200) == pytest.approx(0.1)


def test_k_medoids_recovers_blocks():
    blocks = np.repeat(np.arange(3), 4)
    d = np.where(blocks[:, None] == blocks[None, :], 0.1, 1.0)
    np.fill_diagonal(d, 0.0)
    _, labels, _ = k_medoids(d, 3, seed=1)
    for b in range(3):
        assert len(set(labels[blocks == b])) == 1
    assert len(set(labels)) == 3


@pytest.fixture(scope="module")
def walk_groups(tmp_path_factory):
    from myosuite import make_env

    env = make_env("myoLegWalk-v0")
    model = env.unwrapped.model
    names = [model.actuator(i).name for i in range(model.nu)]
    groups, meta = synergy_groups(model, np.arange(model.nu), 40, steps=3000)
    env.close()
    path = tmp_path_factory.mktemp("dynsyn") / "walk.json"
    DynSynGroups(groups=groups, muscle_names=names, meta=meta).save(path)
    return path, groups


def test_synergy_groups_partition_the_muscles(walk_groups):
    _, groups = walk_groups
    assert len(groups) == 40
    assert sorted(i for g in groups for i in g) == list(range(80))


def test_cpu_wrapper_steps(walk_groups):
    from myosuite import make_env
    from myosuite.envs.wrappers import DynSynWrapper

    path, groups = walk_groups
    env = DynSynWrapper(
        make_env("myoLegWalk-v0"), str(path), start_steps=1, ramp_steps=2
    )
    env.reset(seed=0)
    a = np.full(env.action_space.shape, 0.3, dtype=np.float32)
    assert env.action(a)[groups[0][1:]].tolist() == [pytest.approx(0.3)] * (
        len(groups[0]) - 1
    )
    for _ in range(3):
        env.step(a)
    assert env.amplitude == pytest.approx(0.1)
    env.close()


def test_mjlab_action_term(walk_groups):
    torch = pytest.importorskip("torch")
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import myosuite.envs.myo.backends.mjlab  # noqa: F401
    from myosuite.envs.myo.backends.mjlab.dynsyn_action import apply_dynsyn

    path, groups = walk_groups
    cfg = load_env_cfg("myoLegWalk-v0")
    cfg.scene.num_envs = 2
    apply_dynsyn(cfg, DynSynGroups.load(path), start_steps=0, ramp_steps=1)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    actions = torch.full((2, 80), 0.3)
    env.step(actions)  # the bound is still 0 here; 2 env steps later it is full
    env.step(actions)
    term = env.action_manager.get_term("muscles")
    assert term.action_dim == 80 and term.amplitude == pytest.approx(0.1)
    expected = expand_actions(actions, groups, 0.1, torch)
    torch.testing.assert_close(term.processed_action, expected)
    env.close()


def test_groups_only_reduces_the_action_space(walk_groups):
    from myosuite import make_env
    from myosuite.envs.wrappers import DynSynWrapper

    path, groups = walk_groups
    env = DynSynWrapper(make_env("myoLegWalk-v0"), str(path), 0, 0, max_amplitude=0)
    assert env.action_space.shape == (len(groups),)
    env.reset(seed=0)
    env.step(env.action_space.sample())
    env.close()


def _walk_cfg(num_envs: int):
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    import myosuite.envs.myo.backends.mjlab  # noqa: F401

    cfg = load_env_cfg("myoLegWalk-v0")
    cfg.scene.num_envs = num_envs
    return cfg


def _write_run(run: Path, cfg, counter: int, n_actions: int = 80) -> Path:
    """A run directory as ``train_mjlab.py`` leaves it, with a small actor."""
    torch = pytest.importorskip("torch")
    from mjlab.utils.os import dump_yaml
    from rsl_rl.modules import MLP

    dump_yaml(run / "params" / "env.yaml", asdict(cfg))
    actor = {
        f"mlp.{k}": v for k, v in MLP(4, n_actions, [8], "elu").state_dict().items()
    }
    run.mkdir(exist_ok=True)
    checkpoint = run / "model_1.pt"
    torch.save(
        {
            "actor_state_dict": actor,
            "infos": {"env_state": {"common_step_counter": counter}},
        },
        checkpoint,
    )
    return checkpoint


@pytest.fixture
def trained_run(walk_groups, tmp_path) -> Path:
    """Checkpoint of a 2-env DynSyn run after 100 env steps: bound 0.1 * 200 / 1000."""
    from myosuite.envs.myo.backends.mjlab.dynsyn_action import apply_dynsyn

    cfg = _walk_cfg(2)
    apply_dynsyn(cfg, DynSynGroups.load(walk_groups[0]), 0, 1000)
    return _write_run(tmp_path / "run", cfg, counter=100)


def test_run_directory_records_groups_schedule_and_bound(walk_groups, trained_run):
    from myosuite.utils.rslrl_policy import trained_dynsyn

    trained = trained_dynsyn(trained_run)
    assert trained.groups.groups == walk_groups[1]
    assert (trained.start_steps, trained.ramp_steps) == (0, 1000)
    assert trained.env_steps == 200
    assert trained.amplitude == pytest.approx(0.02)


def test_plain_policy_load_of_a_dynsyn_run_fails(trained_run, tmp_path):
    from myosuite.utils.rslrl_policy import load_rslrl_policy, trained_dynsyn

    with pytest.raises(ValueError, match="trained with DynSyn"):
        load_rslrl_policy(trained_run, 80)
    assert load_rslrl_policy(trained_run, 80, dynsyn_applied=True).action_dim == 80
    plain = _write_run(tmp_path / "plain", _walk_cfg(2), counter=100)
    assert trained_dynsyn(plain) is None
    assert load_rslrl_policy(plain, 80).action_dim == 80


def test_old_run_without_step_offset_is_read(walk_groups, trained_run):
    from myosuite.utils.rslrl_policy import trained_dynsyn

    params = trained_run.parent / "params" / "env.yaml"
    params.write_text(re.sub(r" *step_offset: .*\n", "", params.read_text()))
    assert trained_dynsyn(trained_run).env_steps == 200


def test_evaluation_keeps_the_trained_bound_on_both_backends(trained_run):
    torch = pytest.importorskip("torch")
    from mjlab.envs import ManagerBasedRlEnv

    from myosuite import make_env
    from myosuite.envs.myo.backends.mjlab.dynsyn_action import (
        apply_trained_dynsyn,
        dynsyn_term,
    )
    from myosuite.envs.wrappers import DynSynWrapper
    from myosuite.utils.rslrl_policy import trained_dynsyn

    trained = trained_dynsyn(trained_run)
    cfg = _walk_cfg(1)
    apply_trained_dynsyn(cfg, trained, frozen=True)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    term = dynsyn_term(env)
    assert term.amplitude == pytest.approx(trained.amplitude)
    env.step(torch.full((1, 80), 0.3))
    assert term.amplitude == pytest.approx(trained.amplitude)
    env.close()

    cpu = DynSynWrapper(
        make_env("myoLegWalk-v0"),
        trained.groups,
        trained.start_steps,
        trained.ramp_steps,
        trained.max_amplitude,
        step_offset=trained.env_steps,
        frozen=True,
    )
    cpu.reset(seed=0)
    cpu.step(np.full(80, 0.3, dtype=np.float32))
    assert cpu.amplitude == pytest.approx(trained.amplitude)
    cpu.close()


@pytest.mark.parametrize("num_envs", [2, 1])
def test_resume_continues_the_schedule(trained_run, num_envs):
    torch = pytest.importorskip("torch")
    from mjlab.envs import ManagerBasedRlEnv

    from myosuite.envs.myo.backends.mjlab.dynsyn_action import (
        apply_trained_dynsyn,
        dynsyn_term,
    )
    from myosuite.utils.rslrl_policy import trained_dynsyn

    trained = trained_dynsyn(trained_run)
    cfg = _walk_cfg(num_envs)
    apply_trained_dynsyn(cfg, trained)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    env.common_step_counter = trained.saved_counter  # as the runner's load() does
    term = dynsyn_term(env)
    assert term.env_steps == trained.env_steps
    env.step(torch.full((num_envs, 80), 0.3))
    assert term.env_steps == trained.env_steps + num_envs
    env.close()


def test_success_check_env_follows_the_training_bound(walk_groups):
    pytest.importorskip("mjlab")
    from mjlab.envs import ManagerBasedRlEnv

    from myosuite.envs.myo.backends.mjlab.dynsyn_action import (
        apply_dynsyn,
        dynsyn_term,
    )
    from scripts.train_mjlab import _make_eval_env

    cfg = _walk_cfg(1)
    apply_dynsyn(cfg, DynSynGroups.load(walk_groups[0]), 0, 1000)
    cfg.seed = 0
    train_env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    train_env.common_step_counter = 500
    wrapped = _make_eval_env(
        SimpleNamespace(env=cfg, agent=SimpleNamespace(clip_actions=None)),
        "cpu",
        train_env,
    )
    assert dynsyn_term(wrapped.unwrapped).amplitude == pytest.approx(0.05)
    wrapped.close()
    train_env.close()


def test_resume_refuses_contradicting_groups(walk_groups, trained_run, tmp_path):
    from scripts.train_mjlab import _resolve_dynsyn

    other = DynSynGroups.load(walk_groups[0])
    other.groups = other.groups[::-1]
    other_path = tmp_path / "other.json"
    other.save(other_path)
    cfg = SimpleNamespace(env=_walk_cfg(2), dynsyn_groups=str(other_path))
    with pytest.raises(ValueError, match="differs from the groups"):
        _resolve_dynsyn(cfg, trained_run, 0)

    plain = _write_run(tmp_path / "plain", _walk_cfg(2), counter=100)
    with pytest.raises(ValueError, match="trained without DynSyn"):
        _resolve_dynsyn(cfg, plain, 0)


@pytest.mark.parametrize("task", ["myoElbowPose1D6MRandom-v0", "myoLegWalk-v0"])
def test_both_backends_command_the_same_muscles(task):
    """Same DynSyn action, same muscle commands; the sigmoid comes after the correction."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from myosuite import make_env
    from myosuite.envs.myo.backends.mjlab.dynsyn_action import (
        action_actuator_names,
        apply_dynsyn,
        dynsyn_term,
    )
    from myosuite.envs.wrappers import DynSynWrapper
    from myosuite.terms.base_action import sigmoid_muscle_activation

    cpu = make_env(task)
    model = cpu.unwrapped.model
    names = [model.actuator(i).name for i in range(model.nu)]
    groups = DynSynGroups(
        groups=[list(range(3)), list(range(3, model.nu))], muscle_names=names
    )
    cpu = DynSynWrapper(cpu, groups, 0, 0, 0.2, step_offset=1, frozen=True)
    cpu.reset(seed=0)
    action = np.full(model.nu, 2.0, dtype=np.float32)  # saturated corrections
    action[:2] = [0.4, 0.2]
    cpu.step(action)

    cfg = load_env_cfg(task)
    cfg.scene.num_envs = 1
    sigmoid = cfg.actions["muscles"].muscle_sigmoid
    apply_dynsyn(cfg, groups, 0, 0, 0.2, step_offset=1, frozen=True)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    env.step(torch.as_tensor(action)[None])
    term = dynsyn_term(env)
    assert action_actuator_names(term, env) == names
    twin = term.processed_action[0].numpy()
    np.testing.assert_allclose(twin, cpu.unwrapped.data.ctrl, atol=1e-5)

    raw = expand_actions(action[None], groups.groups, 0.2, np, kappa=0.2)[0]
    expected = sigmoid_muscle_activation(raw, np) if sigmoid else raw
    np.testing.assert_allclose(twin, expected, atol=1e-5)
    env.close()
    cpu.close()


def test_similarity_measures_length_changes_not_slow_drift() -> None:
    """Muscles that only share a slow drift within segments must not look alike."""
    rng = np.random.default_rng(0)
    steps, segment = 2000, 100
    drift = np.linspace(0.0, 0.01, segment)
    a = np.concatenate(
        [
            0.3 + rng.uniform() * 0.1 + drift + 1e-3 * rng.standard_normal(segment)
            for _ in range(20)
        ]
    )
    b = np.concatenate(
        [
            0.3 + rng.uniform() * 0.1 - drift + 1e-3 * rng.standard_normal(segment)
            for _ in range(20)
        ]
    )
    together = a + 0.5
    noise_free = np.stack([a, b, together], axis=1)[:steps]
    similarity = length_similarity(noise_free, segment=segment)
    assert similarity[0, 2] == pytest.approx(1.0)
    assert similarity[0, 1] < 0.05
    # A muscle that doesn't move has no correlation with anything, and no NaN.
    still = np.stack([a, np.full(len(a), 0.3)], axis=1)
    similarity = length_similarity(still, segment=segment)
    assert np.isfinite(similarity).all()
    assert similarity[1, 1] == 1.0
