"""DynSyn muscle synergies: grouping, action mapping, CPU wrapper and mjlab action term."""

import numpy as np
import pytest

from myosuite.core.dynsyn import (
    DynSynGroups,
    expand_actions,
    k_medoids,
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
    if not torch.cuda.is_available():
        pytest.skip("mjlab needs a GPU here")
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import myosuite.envs.myo.backends.mjlab  # noqa: F401
    from myosuite.envs.myo.backends.mjlab.dynsyn_action import apply_dynsyn

    path, groups = walk_groups
    cfg = load_env_cfg("myoLegWalk-v0")
    cfg.scene.num_envs = 2
    apply_dynsyn(cfg, DynSynGroups.load(path), start_steps=0, ramp_steps=1)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env.reset()
    env.step(torch.full((2, 80), 0.3, device="cuda:0"))
    term = env.action_manager.get_term("muscles")
    assert term.action_dim == 80 and term.amplitude == pytest.approx(0.1)
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
