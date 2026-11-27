# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests for packaging metadata, wrapper pickling, and core helpers."""

from __future__ import annotations

import pickle
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import mujoco
import numpy as np
import pytest

from myosuite.core.model_builder import build_from_recipe
from myosuite.core.muscle_conditions import (
    apply_sarcopenia_to_model,
    apply_sarcopenia_to_spec,
)
from myosuite.core.trajectory_io import load_motion_clip, motion_clip_from_states
from myosuite.envs.tasks.mimic.tracking.env import MimicEnv
from myosuite.envs.tasks.fullbody.waypoint.env import WaypointEnv, WaypointTaskCfg
from myosuite.envs.wrappers import FatigueWrapper, MotorNoiseWrapper, SarcopeniaWrapper
from myosuite.terms.base_action import MotorNoiseCfg

pytestmark = pytest.mark.tier1

_HYBRID_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="b" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0" range="-1 1"/>
      <joint name="j2" type="hinge" axis="1 0 0" range="-1 1"/>
      <geom type="sphere" size="0.05" mass="1"/>
      <site name="tip" pos="0 0 0"/>
    </body>
  </worldbody>
  <tendon>
    <fixed name="t1"><joint joint="j1" coef="0.05"/></fixed>
  </tendon>
  <actuator>
    <muscle name="m1" tendon="t1" force="120"/>
    <general name="aff" joint="j2" gaintype="affine" gainprm="10 0 -3.5" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


def test_pyproject_package_data_and_docs_deps() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    pkg_data = cfg["tool"]["setuptools"]["package-data"]["myosuite"]
    excl = cfg["tool"]["setuptools"]["exclude-package-data"]["myosuite"]
    docs_deps = cfg["project"]["optional-dependencies"]["docs"]

    assert "biomechanics/**" in pkg_data
    assert "robot/**" not in pkg_data
    assert "**/.myosuite_resolved_*.xml" in excl
    assert any(dep.startswith("sphinx-autoapi") for dep in docs_deps)


def test_sarcopenia_masks_non_muscle_actuators() -> None:
    spec = mujoco.MjSpec.from_string(_HYBRID_XML)
    apply_sarcopenia_to_spec(spec, force_scale=0.5)
    m_spec = spec.compile()
    assert m_spec.actuator_gainprm[0, 2] == pytest.approx(60.0)
    assert m_spec.actuator_gainprm[1, 2] == pytest.approx(-3.5)

    m_direct = mujoco.MjModel.from_xml_string(_HYBRID_XML)
    apply_sarcopenia_to_model(m_direct, force_scale=0.5)
    assert m_direct.actuator_gainprm[0, 2] == pytest.approx(60.0)
    assert m_direct.actuator_gainprm[1, 2] == pytest.approx(-3.5)


def test_wrapped_waypoint_and_mimic_pickling_preserves_single_scaling() -> None:
    base_model = mujoco.MjModel.from_xml_string(_HYBRID_XML)
    wp = WaypointEnv(
        base_model,
        task=WaypointTaskCfg(site_name="tip", waypoints=((0.2, 0.0),)),
        frame_skip=1,
    )
    wrapped = MotorNoiseWrapper(
        FatigueWrapper(SarcopeniaWrapper(wp, force_scale=0.5)),
        MotorNoiseCfg(signal_dependent_std=0.05),
    )
    restored = pickle.loads(pickle.dumps(wrapped))
    assert restored.unwrapped.model.actuator_gainprm[0, 2] == pytest.approx(60.0)
    assert base_model.actuator_gainprm[0, 2] == pytest.approx(120.0)
    restored.reset(seed=7)
    restored.step(np.zeros(restored.action_space.shape, dtype=np.float32))
    restored.close()
    wrapped.close()

    mimic = MotorNoiseWrapper(
        FatigueWrapper(SarcopeniaWrapper(MimicEnv(frame_skip=10), force_scale=0.5)),
        MotorNoiseCfg(constant_std=0.02),
    )
    f0 = float(mimic.unwrapped.model.actuator_gainprm[0, 2])
    mimic_restored = pickle.loads(pickle.dumps(mimic))
    assert float(
        mimic_restored.unwrapped.model.actuator_gainprm[0, 2]
    ) == pytest.approx(f0)
    mimic_restored.close()
    mimic.close()


def test_hand_pen_rgba_and_musclemimic_bimanual_spec_recipe() -> None:
    pen_model, _ = build_from_recipe("hand_pen")
    sid = mujoco.mj_name2id(pen_model, mujoco.mjtObj.mjOBJ_SITE, "target_ball")
    assert sid >= 0
    assert np.all((pen_model.site_rgba[sid] >= 0.0) & (pen_model.site_rgba[sid] <= 1.0))

    try:
        bim_model, _ = build_from_recipe("musclemimic_bimanual")
    except (FileNotFoundError, ImportError) as exc:
        pytest.skip(f"musclemimic_models unavailable: {exc}")
    assert bim_model.nq > 0


def test_trajectory_io_closes_npz_and_differentiates_in_float64(tmp_path: Path) -> None:
    xml = """
    <mujoco><worldbody><body name="b"><joint name="x" type="slide" axis="1 0 0"/>
    <geom size="0.1"/><site name="s"/></body></worldbody></mujoco>
    """
    m = mujoco.MjModel.from_xml_string(xml)
    dt = 0.01
    qpos = 1000.0 + np.arange(5, dtype=np.float64)[:, None] * 1e-4
    clip = motion_clip_from_states(m, qpos, dt, ("s",))
    assert clip.qvel is not None
    np.testing.assert_allclose(clip.qvel[:, 0], 0.01, rtol=1e-4)

    npz_path = tmp_path / "clip.npz"
    np.savez(npz_path, qpos=qpos)
    loaded = load_motion_clip(npz_path, expected_nq=1, expected_nv=1)
    assert loaded.qpos.shape == (5, 1)
    npz_path.unlink()


def test_pyproject_supports_gymnasium_1_4_and_numpy_2_5() -> None:
    from packaging.requirements import Requirement

    root = Path(__file__).resolve().parents[2]
    cfg = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    deps = {
        Requirement(d).name.lower(): Requirement(d).specifier
        for d in cfg["project"]["dependencies"]
    }

    assert "gymnasium" in deps
    assert deps["gymnasium"].contains("1.2.3")
    assert deps["gymnasium"].contains("1.4.0")
    assert not deps["gymnasium"].contains("2.0.0")

    assert "numpy" in deps
    assert deps["numpy"].contains("2.2.6")
    assert deps["numpy"].contains("2.4.6")
    assert deps["numpy"].contains("2.5.5")


@pytest.mark.skipif(
    __import__("sys").platform == "win32",
    reason="POSIX bash script test",
)
def test_pr_note_and_local_review_shell_syntax_and_fake_gh(tmp_path: Path) -> None:
    """``pr_note.sh`` and ``local_review.sh`` parse cleanly and ``pr_note.sh`` resolves fork/origin PRs in an isolated repo."""
    import os
    import shutil
    import subprocess

    if shutil.which("bash") is None or shutil.which("jq") is None:
        pytest.skip("bash and jq required")

    repo_root = Path(__file__).resolve().parents[2]
    pr_note = repo_root / "scripts" / "review" / "pr_note.sh"
    local_rev = repo_root / "scripts" / "review" / "local_review.sh"
    test_deploy = repo_root / ".github" / "test_deploy.sh"
    for script in (pr_note, local_rev, test_deploy):
        chk = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True
        )
        assert chk.returncode == 0, chk.stderr

    git_dir = tmp_path / "repo"
    git_dir.mkdir()
    subprocess.run(
        ["git", "init", "-b", "test-branch"],
        cwd=git_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=git_dir, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=git_dir, check=True
    )
    (git_dir / "README.md").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=git_dir, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=git_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "remote",
            "add",
            "upstream",
            "https://github.com/staleorg/inaccessible.git",
        ],
        cwd=git_dir,
        check=True,
    )
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/MyoHub/myosuite.git"],
        cwd=git_dir,
        check=True,
    )
    subprocess.run(
        ["git", "remote", "add", "myfork", "https://github.com/forkuser/myosuite.git"],
        cwd=git_dir,
        check=True,
    )
    subprocess.run(
        ["git", "config", "branch.test-branch.remote", "myfork"],
        cwd=git_dir,
        check=True,
    )

    out_print = subprocess.run(
        ["bash", str(pr_note), "print"],
        cwd=git_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "<!-- myo-review:start -->" in out_print.stdout
    assert "Local checks" in out_print.stdout

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    patched_file = tmp_path / "patched_body.txt"
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
jq_expr=""
method="GET"
endpoint=""
body_val=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    api) shift ;;
    --paginate) shift ;;
    -X) method="$2"; shift 2 ;;
    --jq) jq_expr="$2"; shift 2 ;;
    -f) body_val="${{2#body=}}"; shift 2 ;;
    *) endpoint="$1"; shift ;;
  esac
done
if [[ "$method" == "PATCH" ]]; then
  printf '%s' "$body_val" > "{patched_file}"
  echo '{{}}'
  exit 0
fi
if [[ "$endpoint" == "repos/staleorg/inaccessible/"* ]]; then
  echo '{{"message": "Not Found", "status": "404"}}'
  exit 1
elif [[ "$endpoint" == "repos/MyoHub/myosuite/pulls?state=open&head=forkuser%3Atest-branch"* ]]; then
  json='[{{"number": 935, "head": {{"ref": "test-branch", "repo": {{"full_name": "forkuser/myosuite"}}}}}}]'
elif [[ "$endpoint" == *"/pulls?state=open"* ]]; then
  json='[]'
elif [[ "$endpoint" == "repos/MyoHub/myosuite/pulls/935" ]]; then
  json='{{"body": "## Summary\\nHello PR\\n\\n<!-- myo-review:start -->\\nold\\n<!-- myo-review:end -->"}}'
elif [[ "$endpoint" == "repos/forkuser/myosuite" ]]; then
  json='{{"full_name": "forkuser/myosuite", "parent": {{"full_name": "MyoHub/myosuite"}}}}'
else
  json='{{"full_name": "MyoHub/myosuite"}}'
fi
if [[ -n "$jq_expr" ]]; then
  jq -r "$jq_expr" <<<"$json"
else
  echo "$json"
fi
""",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)

    env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}
    body_res = subprocess.run(
        ["bash", str(pr_note), "body", "test-branch"],
        cwd=git_dir,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "## Summary" in body_res.stdout
    assert "Hello PR" in body_res.stdout
    assert "<!-- myo-review:start -->" not in body_res.stdout

    upd_res = subprocess.run(
        ["bash", str(pr_note), "update", "HEAD", "test-branch"],
        cwd=git_dir,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "updated the Local checks block of PR #935" in upd_res.stderr
    patched = patched_file.read_text(encoding="utf-8")
    assert "Hello PR" in patched and "**Local checks**" in patched

    # --record-stdin rejects symbolic --head HEAD and missing --via
    sym_res = subprocess.run(
        ["bash", str(local_rev), "--record-stdin", "--head", "HEAD", "--via", "claude"],
        cwd=git_dir,
        capture_output=True,
        text=True,
    )
    assert sym_res.returncode == 2
    no_via_res = subprocess.run(
        ["bash", str(local_rev), "--record-stdin", "--head", "abcdef123456"],
        cwd=git_dir,
        capture_output=True,
        text=True,
    )
    assert no_via_res.returncode == 2

    # --print-prompt delivers commit log, unified diff hunks, and a readable diff file for read-only IDE reviewers
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=git_dir,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    (git_dir / "README.md").write_text("hi\nsecond line\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=git_dir, check=True)
    subprocess.run(
        ["git", "commit", "-m", "feat: add second line\n\nDetailed body."],
        cwd=git_dir,
        check=True,
        capture_output=True,
    )
    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=git_dir,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    prompt_res = subprocess.run(
        ["bash", str(local_rev), "--print-prompt"],
        cwd=git_dir,
        env={**env, "MYO_REVIEW_BASE": base_sha},
        capture_output=True,
        text=True,
        check=True,
    )
    assert head_sha in prompt_res.stderr
    assert "feat: add second line" in prompt_res.stdout
    assert "+second line" in prompt_res.stdout
    diff_path = git_dir / ".git" / "myo-review" / f"prompt-{head_sha}.diff"
    assert str(diff_path) in prompt_res.stdout
    assert diff_path.is_file()
    assert "+second line" in diff_path.read_text(encoding="utf-8")
