# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Verify MyoSuite's minimal base dependency contract and optional-extra guards."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement
import pytest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

pytestmark = pytest.mark.tier1

_REPO = Path(__file__).resolve().parents[2]

_APPROVED_CORE_DEPENDENCIES = frozenset(
    {
        "numpy",
        "scipy",
        "mujoco",
        "myo-sim",
        "gymnasium",
        "huggingface-hub",
        "filelock",
        "click",
        "termcolor",
        "ml-collections",
        "colorednoise",
    }
)

_EXPECTED_EXTRAS = frozenset(
    {
        "biomechanics",
        "capture",
        "onnx",
        "viz",
        "all",
        "examples",
        "rl",
        "mjlab",
        "mjx",
        "mjx-cuda",
        "docs",
        "musclemimic",
        "dev",
    }
)

_MASKED_OPTIONAL_MODULES = (
    "dm_control",
    "ezc3d",
    "flatten_dict",
    "h5py",
    "imageio",
    "joblib",
    "loop_rate_limiters",
    "matplotlib",
    "mink",
    "mjviser",
    "musclemimic_models",
    "onnx",
    "onnxruntime",
    "osqp",
    "pandas",
    "PIL",
    "sklearn",
    "stable_baselines3",
    "torch",
)


def _norm(name: str) -> str:
    return name.lower().replace("_", "-")


def test_pyproject_declares_minimal_core_dependencies_and_complete_extras() -> None:
    """Base ``project.dependencies`` stays limited to lightweight core runtime packages."""
    cfg = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    raw_deps = cfg["project"]["dependencies"]
    parsed = {_norm(Requirement(raw).name) for raw in raw_deps}
    assert parsed == _APPROVED_CORE_DEPENDENCIES

    opt = cfg["project"]["optional-dependencies"]
    assert _EXPECTED_EXTRAS.issubset(opt.keys())
    viz_names = {_norm(Requirement(r).name) for r in opt["viz"]}
    assert {
        "imageio",
        "h5py",
        "matplotlib",
        "pillow",
        "mjviser",
        "loop-rate-limiters",
    } <= viz_names
    ex_reqs = [Requirement(r) for r in opt["examples"]]
    ex_names = {_norm(req.name) for req in ex_reqs}
    ex_self_extras = {
        extra
        for req in ex_reqs
        if _norm(req.name) == "myosuite"
        for extra in req.extras
    }
    assert {"myosuite", "pandas", "scikit-learn", "joblib"} <= ex_names
    assert {"biomechanics", "rl", "viz", "capture", "onnx"} <= ex_self_extras

    rl_reqs = [Requirement(raw) for raw in opt["rl"]]
    rl_names = {_norm(req.name) for req in rl_reqs}
    rl_self_extras = {
        extra
        for req in rl_reqs
        if _norm(req.name) == "myosuite"
        for extra in req.extras
    }
    assert "onnx" in rl_names
    assert "onnx" in rl_self_extras

    capture_reqs = [
        req
        for req in ex_reqs
        if _norm(req.name) == "myosuite" and "capture" in req.extras
    ]
    assert len(capture_reqs) == 1 and capture_reqs[0].marker is not None
    marker = capture_reqs[0].marker
    assert marker.evaluate(
        {
            "python_version": "3.11",
            "sys_platform": "linux",
            "platform_machine": "x86_64",
        }
    )
    assert marker.evaluate(
        {
            "python_version": "3.11",
            "sys_platform": "darwin",
            "platform_machine": "arm64",
        }
    )
    assert marker.evaluate(
        {"python_version": "3.11", "sys_platform": "win32", "platform_machine": "AMD64"}
    )
    assert not marker.evaluate(
        {
            "python_version": "3.11",
            "sys_platform": "linux",
            "platform_machine": "aarch64",
        }
    )
    assert not marker.evaluate(
        {"python_version": "3.11", "sys_platform": "win32", "platform_machine": "ARM64"}
    )
    assert not marker.evaluate(
        {
            "python_version": "3.15",
            "sys_platform": "linux",
            "platform_machine": "x86_64",
        }
    )

    # Every extra that pulls Pillow directly or via imageio/matplotlib enforces the patched >=13.0.0 floor on Python >=3.11.
    for extra_with_pillow in ("viz", "rl", "mjlab", "mjx"):
        pillow_reqs = [
            Requirement(r)
            for r in opt[extra_with_pillow]
            if _norm(Requirement(r).name) == "pillow"
            and (
                Requirement(r).marker is None
                or Requirement(r).marker.evaluate(
                    {"python_version": "3.11", "sys_platform": "linux"}
                )
            )
        ]
        assert (
            len(pillow_reqs) == 1 and not pillow_reqs[0].specifier.contains("12.3.0")
        ), f"Extra [{extra_with_pillow}] missing Pillow>=13.0.0 floor on Python 3.11 Linux"


def test_core_suites_run_when_all_optional_extras_are_absent() -> None:
    """Fresh Python subprocess with all 19 optional/removed packages blocked runs core CPU envs."""
    assert len(_MASKED_OPTIONAL_MODULES) == 19
    code = f"""
import importlib.abc, sys
BLOCKED = set({_MASKED_OPTIONAL_MODULES!r})

class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"blocked optional dependency: {{fullname}}", name=fullname)
        return None

sys.meta_path.insert(0, _Blocker())
import numpy as np
import myosuite
from myosuite import make_env, register_all_envs
from myosuite.viz.grouped_datasets import Trace

register_all_envs()
for env_id in (
    "myoElbowPose1D6MRandom-v0",
    "myoChallengeSoccerP1-v0",
    "myoMimicBimanual-v0",
):
    env = make_env(env_id)
    try:
        obs, _ = env.reset(seed=0)
        obs2, rew, term, trunc, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert np.isfinite(obs).all() and np.isfinite(obs2).all() and np.isfinite(rew)
    finally:
        env.close()

# In-memory/pickle Trace works without h5py or PIL installed
tr = Trace("probe")
tr.create_group("Trial0")
tr.append_datum("Trial0", "time", np.array(0.0))
tr.close()
assert "Trial0" in repr(tr)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"Core CPU envs failed when optional extras were blocked:\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


def test_optional_extra_guards_report_install_commands_and_reraise_internal_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional biomechanics/viz/capture/musclemimic guards name the extra and re-raise broken internal deps."""
    from myosuite.core.import_utils import require_extra
    import importlib

    def _broken_import(name: str):
        raise ModuleNotFoundError(
            "No module named '_broken_c_ext'", name="_broken_c_ext"
        )

    monkeypatch.setattr(importlib, "import_module", _broken_import)
    with pytest.raises(ModuleNotFoundError, match="_broken_c_ext"):
        require_extra("matplotlib.pyplot", "viz", "plot")
    monkeypatch.undo()

    code = """
import importlib.abc, sys
BLOCKED = {"mink", "osqp", "imageio", "PIL", "h5py", "matplotlib", "ezc3d", "sklearn", "joblib"}

class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"blocked: {fullname}", name=fullname)
        return None

sys.meta_path.insert(0, _Blocker())
import mujoco, numpy as np
from pathlib import Path
from myosuite.biomechanics import inverse_kinematics as ik
from myosuite.biomechanics import static_optimization as so
from myosuite.viz import video_io, grouped_datasets, path_plotting
from myosuite.integrations.capture import markers
from myosuite.integrations.musclemimic import sar_extraction

xml = "<mujoco><worldbody><body name='b'><joint name='j'/><geom size='0.1'/><site name='s'/></body></worldbody><actuator><motor joint='j'/></actuator></mujoco>"
m = mujoco.MjModel.from_xml_string(xml)

checks = [
    (lambda: ik.solve_ik(m, {"s": np.zeros(3)}), "myosuite[biomechanics]"),
    (lambda: so.static_optimization(m, np.zeros((2, 1)), np.zeros((2, 1)), np.zeros((2, 1))), "myosuite[biomechanics]"),
    (lambda: video_io.write_video("out.mp4", [np.zeros((8, 8, 3), dtype=np.uint8)]), "myosuite[viz]"),
    (lambda: path_plotting.plot({}, None), "myosuite[viz]"),
    (lambda: grouped_datasets.Trace.load("nonexistent.h5"), "myosuite[viz]"),
    (lambda: markers._read_c3d(Path("nonexistent.c3d")), "myosuite[capture]"),
    (lambda: sar_extraction._import_sklearn_pca(), "myosuite[musclemimic]"),
    (lambda: sar_extraction.load_synergy_model("nonexistent_dir"), "myosuite[musclemimic]"),
]
for fn, expected in checks:
    try:
        fn()
    except ImportError as exc:
        assert expected in str(exc), f"Expected {expected!r} in {exc!r}"
    else:
        raise AssertionError(f"Expected ImportError containing {expected!r}")
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"


def test_examine_env_validates_viz_extras_before_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``examine_env`` checks ``h5py``/``matplotlib``/``imageio`` before running any rollout steps."""
    from click.testing import CliRunner
    from myosuite.benchmarks.examine_env import main as examine_main
    import myosuite.benchmarks.examine_env as examine_mod

    called = False

    def _forbid_make_env(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError(
            "make_env must not be called when required [viz] dep is missing"
        )

    monkeypatch.setattr(examine_mod, "make_env", _forbid_make_env)
    monkeypatch.setitem(sys.modules, "h5py", None)
    runner = CliRunner()
    res = runner.invoke(
        examine_main,
        [
            "-e",
            "myoElbowPose1D6MRandom-v0",
            "-n",
            "1",
            "-r",
            "none",
            "--save_paths",
            "True",
        ],
    )
    assert res.exit_code != 0
    assert isinstance(res.exception, ImportError)
    assert "myosuite[viz]" in str(res.exception)
    assert not called
