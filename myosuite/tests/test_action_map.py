"""Action maps (DynSyn): mapping, CPU wrapper, mjlab action term, checkpoint playback.

Learning the groups and training are tested in ``training/dynsyn``.
"""

import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from myosuite.utils.action_map import (
    ActionMap,
    ActionMapWrapper,
    DynSyn,
    DynSynGroups,
    weight_amplitude,
)

pytestmark = pytest.mark.tier2

GROUPS = [[0, 2], [1], [3, 4, 5]]
NAMES = [f"m{i}" for i in range(6)]


def expand_actions(actions, groups, amplitude, xp, kappa=0.1):
    """Per-muscle actions of a DynSyn map with a fixed correction bound."""
    m = DynSyn(DynSynGroups(groups, NAMES[: sum(map(len, groups))]), 0, 1, kappa)
    state = {"env_steps": amplitude / kappa}  # the schedule's bound is then amplitude
    if xp is np:
        return m.forward(np.asarray(actions, dtype=np.float64), state)
    return m.forward_torch(actions, state)


# Outputs of the expand_actions / weight_amplitude functions this map replaced, for
# Z with groups GROUPS, start 100 and ramp 200: (env steps, max amplitude) -> output.
Z = [
    [0.002, 0.597, -0.548, -1.781, -0.909, -1.983],
    [0.12, 2.68, -0.984, -1.241, 0.98, 0.714],
]
REFERENCE = {
    (150, 0.1): [
        [0.002, 0.597, 0.00195, -0.548, -0.5343, -0.5343],
        [0.12, 2.68, 0.11699999999999999, -0.984, -1.0086, -1.0086],
    ],
    (1e9, 0.1): [
        [
            0.002,
            0.597,
            0.0018110394790250108,
            -0.548,
            -0.5085082737102264,
            -0.4952381970009459,
        ],
        [
            0.12,
            2.68,
            0.10985111328548287,
            -0.984,
            -1.0581016850391638,
            -1.0443368138584825,
        ],
    ],
    (1e9, 0.3): [
        [
            0.002,
            0.597,
            0.0014331184370750326,
            -0.548,
            -0.42952482113067914,
            -0.3897145910028374,
        ],
        [
            0.12,
            2.68,
            0.08955333985644863,
            -0.984,
            -1.2063050551174916,
            -1.1650104415754483,
        ],
    ],
}


@pytest.mark.parametrize(("steps", "max_amplitude"), list(REFERENCE))
def test_dynsyn_map_matches_the_replaced_functions(steps, max_amplitude):
    m = DynSyn(DynSynGroups(GROUPS, NAMES), 100, 200, max_amplitude)
    assert isinstance(m, ActionMap) and m.action_dim == m.n_muscles == 6
    out = m.forward(np.array(Z), {"env_steps": steps})
    np.testing.assert_allclose(out, REFERENCE[steps, max_amplitude], rtol=1e-15)
    torch = pytest.importorskip("torch")
    out_t = m.forward_torch(
        torch.tensor(Z, dtype=torch.float64), {"env_steps": steps}
    ).numpy()
    np.testing.assert_allclose(out_t, REFERENCE[steps, max_amplitude], rtol=1e-15)


def test_manifest_round_trip():
    from myosuite.utils.action_map import action_map_from_manifest

    m = DynSyn(DynSynGroups(GROUPS, NAMES, {"seed": 1}), 3.0, 4.0, 0.0)
    back = action_map_from_manifest(m.to_manifest())
    assert back == m and back.action_dim == 3


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


@pytest.fixture(scope="module")
def walk_groups(tmp_path_factory):
    from myosuite import make_env

    env = make_env("myoLegWalk-v0")
    model = env.unwrapped.model
    names = [model.actuator(i).name for i in range(model.nu)]
    env.close()
    # Fixed pairs instead of fitted groups (fitting is tested in training/dynsyn).
    groups = [[2 * i, 2 * i + 1] for i in range(model.nu // 2)]
    path = tmp_path_factory.mktemp("dynsyn") / "walk.json"
    DynSynGroups(groups=groups, muscle_names=names).save(path)
    return path, groups


def test_cpu_wrapper_steps(walk_groups):
    from myosuite import make_env

    path, groups = walk_groups
    env = ActionMapWrapper(
        make_env("myoLegWalk-v0"),
        DynSyn(DynSynGroups.load(path), start_steps=1, ramp_steps=2),
    )
    env.reset(seed=0)
    a = np.full(env.action_space.shape, 0.3, dtype=np.float32)
    assert env.action(a)[groups[0][1:]].tolist() == [pytest.approx(0.3)] * (
        len(groups[0]) - 1
    )
    for _ in range(3):
        env.step(a)
    assert env.action_map.amplitude(env.env_steps) == pytest.approx(0.1)
    env.close()


def test_mjlab_action_term(walk_groups):
    torch = pytest.importorskip("torch")
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import myosuite.envs.myo.backends.mjlab  # noqa: F401
    from myosuite.envs.myo.backends.mjlab.tasks.mdp.action_map import apply_action_map

    path, groups = walk_groups
    cfg = load_env_cfg("myoLegWalk-v0")
    cfg.scene.num_envs = 2
    apply_action_map(cfg, DynSyn(DynSynGroups.load(path), start_steps=0, ramp_steps=1))
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    actions = torch.full((2, 80), 0.3)
    env.step(actions)  # the bound is still 0 here; 2 env steps later it is full
    env.step(actions)
    term = env.action_manager.get_term("muscles")
    assert term.action_dim == 80
    assert term.action_map.amplitude(term.env_steps) == pytest.approx(0.1)
    expected = expand_actions(actions, groups, 0.1, torch)
    torch.testing.assert_close(term.processed_action, expected)
    env.close()


def test_groups_only_reduces_the_action_space(walk_groups):
    from myosuite import make_env

    path, groups = walk_groups
    env = ActionMapWrapper(
        make_env("myoLegWalk-v0"), DynSyn(DynSynGroups.load(path), 0, 0, 0)
    )
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
    pytest.importorskip("mjlab")
    from myosuite.envs.myo.backends.mjlab.tasks.mdp.action_map import apply_action_map

    cfg = _walk_cfg(2)
    apply_action_map(cfg, DynSyn(DynSynGroups.load(walk_groups[0]), 0, 1000))
    return _write_run(tmp_path / "run", cfg, counter=100)


def test_run_directory_records_groups_schedule_and_bound(walk_groups, trained_run):
    from myosuite.utils.action_map import trained_action_map

    trained = trained_action_map(trained_run)
    m = trained.action_map
    assert m.groups.groups == walk_groups[1]
    assert (m.start_steps, m.ramp_steps) == (0, 1000)
    assert trained.env_steps == 200
    assert m.amplitude(trained.env_steps) == pytest.approx(0.02)


def test_plain_policy_load_of_a_dynsyn_run_fails(trained_run, tmp_path):
    from myosuite.utils.action_map import trained_action_map
    from myosuite.utils.rslrl_policy import load_rslrl_policy

    from myosuite.utils.action_map import load_policy

    with pytest.raises(ValueError, match="'dynsyn' action map"):
        load_rslrl_policy(trained_run, 80)
    policy, trained = load_policy(trained_run, 80)
    assert policy.action_dim == 80 and trained.action_map.kind == "dynsyn"
    plain = _write_run(tmp_path / "plain", _walk_cfg(2), counter=100)
    assert trained_action_map(plain) is None
    assert load_rslrl_policy(plain, 80).action_dim == 80


def test_old_run_without_step_offset_is_read(walk_groups, trained_run):
    from myosuite.utils.action_map import trained_action_map

    params = trained_run.parent / "params" / "env.yaml"
    params.write_text(re.sub(r" *step_offset: .*\n", "", params.read_text()))
    assert trained_action_map(trained_run).env_steps == 200


def test_evaluation_keeps_the_trained_bound_on_both_backends(trained_run):
    torch = pytest.importorskip("torch")
    from mjlab.envs import ManagerBasedRlEnv

    from myosuite import make_env
    from myosuite.envs.myo.backends.mjlab.tasks.mdp.action_map import (
        action_map_term,
        apply_trained_action_map,
    )
    from myosuite.utils.action_map import trained_action_map

    trained = trained_action_map(trained_run)
    cfg = _walk_cfg(1)
    apply_trained_action_map(cfg, trained, frozen=True)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    term = action_map_term(env)
    assert term.env_steps == trained.env_steps
    env.step(torch.full((1, 80), 0.3))
    assert term.env_steps == trained.env_steps
    env.close()

    cpu = ActionMapWrapper(
        make_env("myoLegWalk-v0"),
        trained.action_map,
        step_offset=trained.env_steps,
        frozen=True,
    )
    cpu.reset(seed=0)
    cpu.step(np.full(80, 0.3, dtype=np.float32))
    assert cpu.env_steps == trained.env_steps
    cpu.close()


@pytest.mark.parametrize("num_envs", [2, 1])
def test_resume_continues_the_schedule(trained_run, num_envs):
    torch = pytest.importorskip("torch")
    from mjlab.envs import ManagerBasedRlEnv

    from myosuite.envs.myo.backends.mjlab.tasks.mdp.action_map import (
        action_map_term,
        apply_trained_action_map,
    )
    from myosuite.utils.action_map import trained_action_map

    trained = trained_action_map(trained_run)
    cfg = _walk_cfg(num_envs)
    apply_trained_action_map(cfg, trained)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    env.common_step_counter = trained.saved_counter  # as the runner's load() does
    term = action_map_term(env)
    assert term.env_steps == trained.env_steps
    env.step(torch.full((num_envs, 80), 0.3))
    assert term.env_steps == trained.env_steps + num_envs
    env.close()


@pytest.mark.parametrize("task", ["myoElbowPose1D6MRandom-v0", "myoLegWalk-v0"])
def test_both_backends_command_the_same_muscles(task):
    """Same DynSyn action, same muscle commands; the sigmoid comes after the correction."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from myosuite import make_env
    from myosuite.envs.myo.backends.mjlab.tasks.mdp.action_map import (
        action_actuator_names,
        action_map_term,
        apply_action_map,
    )
    from myosuite.terms.base_action import sigmoid_muscle_activation

    cpu = make_env(task)
    model = cpu.unwrapped.model
    names = [model.actuator(i).name for i in range(model.nu)]
    groups = DynSynGroups(
        groups=[list(range(3)), list(range(3, model.nu))], muscle_names=names
    )
    cpu = ActionMapWrapper(cpu, DynSyn(groups, 0, 0, 0.2), step_offset=1, frozen=True)
    cpu.reset(seed=0)
    action = np.full(model.nu, 2.0, dtype=np.float32)  # saturated corrections
    action[:2] = [0.4, 0.2]
    cpu.step(action)

    cfg = load_env_cfg(task)
    cfg.scene.num_envs = 1
    sigmoid = cfg.actions["muscles"].muscle_sigmoid
    apply_action_map(cfg, DynSyn(groups, 0, 0, 0.2), step_offset=1, frozen=True)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    env.step(torch.as_tensor(action)[None])
    term = action_map_term(env)
    assert action_actuator_names(term, env) == names
    twin = term.processed_action[0].numpy()
    np.testing.assert_allclose(twin, cpu.unwrapped.data.ctrl, atol=1e-5)

    raw = expand_actions(action[None], groups.groups, 0.2, np, kappa=0.2)[0]
    expected = sigmoid_muscle_activation(raw, np) if sigmoid else raw
    np.testing.assert_allclose(twin, expected, atol=1e-5)
    env.close()
    cpu.close()


def test_checkpoints_of_the_earlier_dynsyn_format_are_read_and_refused(
    walk_groups, tmp_path
):
    """Checkpoints written before the generic action-map record (``infos["dynsyn"]``,
    or only a ``DynSynActionCfg`` in ``params/env.yaml``), and runs whose env.yaml holds
    the earlier plain-class ``ActionMapActionCfg`` (a ``!!python/object`` tag)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("mjlab")
    from myosuite.utils.action_map import trained_action_map
    from myosuite.utils.rslrl_policy import load_rslrl_policy

    groups = DynSynGroups.load(walk_groups[0])
    legacy = {
        "groups": asdict(groups),
        "start_steps": 0,
        "ramp_steps": 1000,
        "max_amplitude": 0.1,
        "env_steps": 200,
    }
    plain = _write_run(tmp_path / "plain", _walk_cfg(2), counter=100)
    state = torch.load(plain, weights_only=True)
    state["infos"]["dynsyn"] = legacy
    lone = tmp_path / "lone" / "model_1.pt"
    lone.parent.mkdir()
    torch.save(state, lone)

    run = tmp_path / "run" / "model_1.pt"
    run.parent.mkdir()
    shutil.copy(plain, run)
    term = {k: v for k, v in legacy.items() if k != "env_steps"}
    (run.parent / "params").mkdir()
    (run.parent / "params" / "env.yaml").write_text(
        "actions:\n  muscles: !!python/object:myosuite.envs.myo.backends.mjlab."
        "dynsyn_action.DynSynActionCfg\n"
        + "".join(f"    {k}: {v!r}\n" for k, v in term.items() if k != "groups")
        + f"    groups: {json.dumps(term['groups'])}\n"
        + "scene:\n  num_envs: 2\n"
    )
    tagged = tmp_path / "tagged" / "model_1.pt"
    tagged.parent.mkdir()
    shutil.copy(plain, tagged)
    manifest = DynSyn.from_manifest(term).to_manifest()
    (tagged.parent / "params").mkdir()
    (tagged.parent / "params" / "env.yaml").write_text(
        "actions:\n  muscles: !!python/object:myosuite.methods.synergies.base."
        "ActionMapActionCfg\n"
        + f"    action_map: {json.dumps(manifest)}\n"
        + "    frozen: false\n    step_offset: 0.0\n"
        + "scene:\n  num_envs: 2\n"
    )
    for checkpoint in (lone, run, tagged):
        trained = trained_action_map(checkpoint)
        assert trained.action_map.groups.groups == walk_groups[1]
        assert trained.env_steps == 200
        assert trained.action_map.amplitude(200) == pytest.approx(0.02)
        with pytest.raises(ValueError, match="'dynsyn' action map"):
            load_rslrl_policy(checkpoint, 80)


def test_manifest_json_action_map_is_detected(tmp_path):
    from myosuite.utils.checkpoint_manifest import action_map_kind, write_manifest

    m = DynSyn(DynSynGroups(GROUPS, NAMES))
    (tmp_path / "model_1.pt").touch()
    write_manifest(tmp_path, "myoLegWalk-v0", contract={}, action_map=m.to_manifest())
    assert action_map_kind({}, tmp_path / "model_1.pt") == "dynsyn"
    (tmp_path / "manifest.json").unlink()
    assert action_map_kind({}, tmp_path / "model_1.pt") is None
