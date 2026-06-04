#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Generate a consolidated parity report across all refactoring phases.

Runs Phase 1 (term function unit tests), Phase 2 (CPU numerical replay
against frozen baselines), and Phase 3 (MJX vs CPU backend parity), then
writes a human-readable Markdown report to reports/parity_report.md.

This script is a *reporting tool*, not a gating test. It never raises on
divergence; the CI gating is handled by test_parity.py directly.

Usage:
    python scripts/generate_parity_report.py [--output reports/parity_report.md]
"""

from __future__ import annotations

import argparse
import datetime
import pickle
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_TESTS_DIR = _REPO_ROOT / "myosuite" / "tests"
_BASELINE_DIR = _TESTS_DIR / "parity_baselines"
_REPORTS_DIR = _REPO_ROOT / "reports"

_ATOL_PHASE2 = 1e-6

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _status_icon(passed: bool) -> str:
    """Return a checkmark or cross emoji for report tables.

    Args:
        passed: Whether the check passed.

    Returns:
        ``"✅"`` when passed, ``"❌"`` otherwise.
    """
    return "✅" if passed else "❌"


def _fmt_sci(x: float) -> str:
    """Format a float in scientific notation for the report.

    Args:
        x: Value to format.

    Returns:
        String like ``"1.23e-08"`` or ``"< 1e-16"`` for very small values.
    """
    if x == 0.0 or x < 1e-16:
        return "< 1e-16"
    return f"{x:.2e}"


def _git_branch() -> str:
    """Return the current git branch name, or ``"unknown"`` on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _run_pytest(
    test_path: str,
    extra_args: list[str] | None = None,
    use_uv: bool = False,
) -> tuple[int, str]:
    """Run pytest on *test_path* and return exit code + combined output.

    Args:
        test_path: Path (string) to the test file or directory.
        extra_args: Additional pytest arguments.
        use_uv: If True, invoke via ``uv run pytest`` instead of the current
            Python interpreter. Used for Phase 3 which needs the uv env.

    Returns:
        Tuple ``(returncode, output_text)``.
    """
    args = extra_args or []
    cmd = (
        ["uv", "run", "pytest"] + [test_path] + args + ["--tb=short", "-q"]
        if use_uv
        else [sys.executable, "-m", "pytest"]
        + [test_path]
        + args
        + ["--tb=short", "-q"]
    )
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    return result.returncode, result.stdout + result.stderr


def _parse_pytest_summary(output: str) -> tuple[int, int]:
    """Extract (passed, failed) counts from pytest -q output.

    Args:
        output: Combined stdout/stderr from a pytest -q run.

    Returns:
        Tuple ``(passed, failed)``. Returns ``(0, 0)`` if parsing fails.
    """
    # pytest -q outputs lines like "7 passed", "3 passed, 1 failed", "1 error"
    passed = 0
    failed = 0
    for line in output.splitlines():
        m_pass = re.search(r"(\d+) passed", line)
        m_fail = re.search(r"(\d+) failed", line)
        m_err = re.search(r"(\d+) error", line)
        if m_pass:
            passed = int(m_pass.group(1))
        if m_fail:
            failed = int(m_fail.group(1))
        if m_err:
            failed += int(m_err.group(1))
    return passed, failed


# ---------------------------------------------------------------------------
# Phase 2 — CPU baseline replay
# ---------------------------------------------------------------------------


def _first_fail_step(obs_arr: np.ndarray, rwd_diffs: list[float]) -> int | None:
    """Return the first step index where any tolerance is exceeded.

    Args:
        obs_arr: Shape ``(T, obs_dim)`` array of absolute obs differences.
        rwd_diffs: List of absolute reward differences, length T.

    Returns:
        Zero-based step index of first failure, or ``None`` if all pass.
    """
    for i in range(len(rwd_diffs)):
        if obs_arr[i].max() > _ATOL_PHASE2 or rwd_diffs[i] > _ATOL_PHASE2:
            return i
    return None


def _replay_baseline(env_id: str, pkl_path: Path) -> dict:
    """Replay a frozen baseline and compute per-step divergence statistics.

    Mirrors the logic in ``test_parity.py`` but collects diffs instead of
    asserting, so the report can show how close the numbers are.

    Args:
        env_id: Gymnasium environment identifier.
        pkl_path: Path to the ``.pkl`` baseline file.

    Returns:
        Dict with keys: ``n_steps``, ``obs_max``, ``obs_mean``, ``rwd_max``,
        ``passed``, ``first_fail_step``, ``error`` (or ``None``).
    """
    import myosuite

    myosuite.register_all_envs()
    from myosuite.utils import gym

    try:
        with open(pkl_path, "rb") as fh:
            records: list[dict] = pickle.load(fh)
    except Exception as exc:
        return {"error": f"Cannot load baseline: {exc}", "passed": False}

    try:
        env = gym.make(env_id)
    except Exception as exc:
        return {"error": f"Cannot instantiate env: {exc}", "passed": False}

    obs_diffs: list[np.ndarray] = []
    rwd_diffs: list[float] = []
    current_episode_np_state = None

    try:
        env.reset()

        for rec in records:
            if rec["episode_np_state"] is not current_episode_np_state:
                current_episode_np_state = rec["episode_np_state"]
                env.unwrapped.np_random.bit_generator.state = current_episode_np_state
                env.reset()

            obs, rwd, _terminated, _truncated, _ = env.step(rec["action"])
            obs_diffs.append(np.abs(np.asarray(obs, dtype=np.float64) - rec["obs"]))
            rwd_diffs.append(abs(float(rwd) - float(rec["rwd"])))

    except Exception as exc:
        env.close()
        return {"error": f"Replay error: {exc}", "passed": False}
    finally:
        env.close()

    if not obs_diffs:
        return {"error": "No steps replayed", "passed": False}

    obs_arr = np.array(obs_diffs)  # (T, obs_dim)
    obs_max = float(obs_arr.max())
    obs_mean = float(obs_arr.mean())
    rwd_max = float(max(rwd_diffs))
    passed = obs_max <= _ATOL_PHASE2 and rwd_max <= _ATOL_PHASE2

    return {
        "n_steps": len(records),
        "obs_max": obs_max,
        "obs_mean": obs_mean,
        "rwd_max": rwd_max,
        "passed": passed,
        "first_fail_step": _first_fail_step(obs_arr, rwd_diffs),
        "error": None,
    }


# ---------------------------------------------------------------------------
# Report section builders
# ---------------------------------------------------------------------------


def _build_header(branch: str) -> str:
    """Build the report header with timestamp and environment metadata.

    Args:
        branch: Current git branch name.

    Returns:
        Markdown string for the report header.
    """
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    py_version = platform.python_version()
    return (
        f"# MyoSuite Parity Report\n\n"
        f"Generated: {ts}  \n"
        f"Branch: `{branch}`  \n"
        f"Python: {py_version}\n"
    )


def _build_summary(
    phase1_passed: int,
    phase1_failed: int,
    phase2_results: list[dict],
    phase3_passed: int,
    phase3_failed: int,
    phase3_skipped: bool,
) -> str:
    """Build the summary table for all phases.

    Args:
        phase1_passed: Number of Phase 1 tests that passed.
        phase1_failed: Number of Phase 1 tests that failed.
        phase2_results: List of per-env result dicts from Phase 2 replay.
        phase3_passed: Number of Phase 3 tests that passed.
        phase3_failed: Number of Phase 3 tests that failed.
        phase3_skipped: True if Phase 3 was skipped due to missing MJX.

    Returns:
        Markdown string for the summary section.
    """
    p1_total = phase1_passed + phase1_failed
    p1_ok = phase1_failed == 0
    p1_status = _status_icon(p1_ok)

    p2_n = len(phase2_results)
    p2_pass = sum(1 for _, r in phase2_results if r.get("passed", False))
    p2_fail = p2_n - p2_pass
    p2_ok = p2_fail == 0
    p2_status = _status_icon(p2_ok)

    if phase3_skipped:
        p3_row = "| 3 | MJX/XLA vs CPU | — | — | — | ⏭ Skipped |"
    else:
        p3_total = phase3_passed + phase3_failed
        p3_ok = phase3_failed == 0
        p3_status = _status_icon(p3_ok)
        p3_row = f"| 3 | MJX/XLA vs CPU | {p3_total} | {phase3_passed} | {phase3_failed} | {p3_status} |"

    lines = [
        "## Summary",
        "",
        "| Phase | Description | Tests | Passed | Failed | Status |",
        "|-------|-------------|-------|--------|--------|--------|",
        f"| 1 | Term functions (CPU) | {p1_total} | {phase1_passed} | {phase1_failed} | {p1_status} |",
        f"| 2 | CPU numerical replay | {p2_n} | {p2_pass} | {p2_fail} | {p2_status} |",
        p3_row,
    ]
    return "\n".join(lines)


def _build_phase1_section(
    returncode: int, output: str, passed: int, failed: int
) -> str:
    """Build the Phase 1 section for the report.

    Args:
        returncode: Pytest exit code.
        output: Combined stdout/stderr from pytest.
        passed: Number of passing tests.
        failed: Number of failing tests.

    Returns:
        Markdown string for the Phase 1 section.
    """
    status = _status_icon(returncode == 0)
    lines = [
        "---",
        "",
        "## Phase 1 — Term Function Tests",
        "",
        f"**Status**: {status}  |  Passed: {passed}  |  Failed: {failed}",
        "",
        "```",
        output.strip(),
        "```",
    ]
    return "\n".join(lines)


def _build_phase2_section(results: list[tuple[str, dict]]) -> str:
    """Build the Phase 2 section with per-env divergence tables.

    Args:
        results: List of ``(env_id, stats_dict)`` pairs.

    Returns:
        Markdown string for the Phase 2 section.
    """
    lines = [
        "---",
        "",
        "## Phase 2 — CPU Numerical Replay",
        "",
        f"Tolerance: `atol = {_ATOL_PHASE2:.0e}`",
        "",
    ]

    for env_id, stats in results:
        status = _status_icon(stats.get("passed", False))
        lines.append(f"### {env_id}")
        lines.append("")

        if stats.get("error"):
            lines.append(f"**Error**: `{stats['error']}`  |  Status: ❌")
            lines.append("")
            continue

        n_steps = stats["n_steps"]
        obs_max = _fmt_sci(stats["obs_max"])
        obs_mean = _fmt_sci(stats["obs_mean"])
        rwd_max = _fmt_sci(stats["rwd_max"])

        lines += [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Steps | {n_steps} |",
            f"| Max \\|obs error\\| | {obs_max} |",
            f"| Mean \\|obs error\\| | {obs_mean} |",
            f"| Max \\|reward error\\| | {rwd_max} |",
            f"| Status | {status} |",
        ]

        if not stats["passed"] and stats["first_fail_step"] is not None:
            lines.append(f"| First failure at step | {stats['first_fail_step']} |")

        lines.append("")

    return "\n".join(lines)


def _build_phase3_section(
    returncode: int,
    output: str,
    passed: int,
    failed: int,
    skipped: bool,
    skip_reason: str,
) -> str:
    """Build the Phase 3 section for the report.

    Args:
        returncode: Pytest exit code (ignored when skipped).
        output: Combined stdout/stderr from pytest.
        passed: Number of passing tests.
        failed: Number of failing tests.
        skipped: True if the entire Phase 3 was skipped.
        skip_reason: Human-readable reason for skipping.

    Returns:
        Markdown string for the Phase 3 section.
    """
    lines = [
        "---",
        "",
        "## Phase 3 — MJX/XLA Backend Parity",
        "",
    ]

    if skipped:
        lines += [
            "**Skipped**: MJX/mujoco_playground not available.",
            "",
            f"> {skip_reason}",
        ]
        return "\n".join(lines)

    status = _status_icon(returncode == 0)
    lines += [
        f"**Status**: {status}  |  Passed: {passed}  |  Failed: {failed}",
        "",
        "```",
        output.strip(),
        "```",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main(output_path: Path) -> None:
    """Generate the full parity report and write it to *output_path*.

    Args:
        output_path: Destination ``.md`` file for the report.
    """
    branch = _git_branch()
    print(f"Branch: {branch}")
    print(f"Output: {output_path}")
    print()

    # -- Phase 1 ----------------------------------------------------------
    print("Phase 1: running term function tests...")
    p1_rc, p1_out = _run_pytest(str(_TESTS_DIR / "test_terms_cpu.py"))
    p1_passed, p1_failed = _parse_pytest_summary(p1_out)
    print(f"  -> {p1_passed} passed, {p1_failed} failed (rc={p1_rc})")

    # -- Phase 2 ----------------------------------------------------------
    print("\nPhase 2: replaying CPU baselines...")
    baselines = sorted(_BASELINE_DIR.glob("*.pkl"))
    phase2_results: list[tuple[str, dict]] = []

    for pkl_path in baselines:
        env_id = pkl_path.stem
        print(f"  replaying {env_id} ...", end=" ", flush=True)
        stats = _replay_baseline(env_id, pkl_path)
        phase2_results.append((env_id, stats))
        icon = _status_icon(stats.get("passed", False))
        if stats.get("error"):
            print(f"{icon}  ERROR: {stats['error']}")
        else:
            print(
                f"{icon}  obs_max={_fmt_sci(stats['obs_max'])}  "
                f"rwd_max={_fmt_sci(stats['rwd_max'])}"
            )

    # -- Phase 3 ----------------------------------------------------------
    print("\nPhase 3: running MJX backend parity tests...")
    uv_bin = shutil.which("uv")
    p3_skip = False
    p3_skip_reason = ""
    p3_rc, p3_out = 0, ""
    p3_passed, p3_failed = 0, 0

    p3_test = str(_TESTS_DIR / "test_backends.py")

    if uv_bin:
        print(f"  using uv at {uv_bin}")
        p3_rc, p3_out = _run_pytest(p3_test, use_uv=True)
    else:
        p3_rc, p3_out = _run_pytest(p3_test, use_uv=False)

    # Detect module-level skip (whole file skipped due to missing MJX)
    if "no tests ran" in p3_out or re.search(r"0 passed.*skipped", p3_out):
        p3_skip = True
        # Extract the skip reason from the output
        m = re.search(r"SKIP.*?['\"](.+?)['\"]", p3_out)
        p3_skip_reason = m.group(1) if m else p3_out.strip().split("\n")[-1]
    else:
        p3_passed, p3_failed = _parse_pytest_summary(p3_out)

    icon = "⏭" if p3_skip else _status_icon(p3_rc == 0)
    print(f"  -> {icon}  passed={p3_passed}  failed={p3_failed}  skipped={p3_skip}")

    # -- Assemble report --------------------------------------------------
    print("\nAssembling report...")
    sections = [
        _build_header(branch),
        "",
        _build_summary(
            p1_passed, p1_failed, phase2_results, p3_passed, p3_failed, p3_skip
        ),
        "",
        _build_phase1_section(p1_rc, p1_out, p1_passed, p1_failed),
        "",
        _build_phase2_section(phase2_results),
        "",
        _build_phase3_section(
            p3_rc, p3_out, p3_passed, p3_failed, p3_skip, p3_skip_reason
        ),
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPORTS_DIR / "parity_report.md",
        help="Output Markdown file (default: reports/parity_report.md)",
    )
    args = parser.parse_args()
    main(args.output)
