# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Locate MuscleMimic for ``python -m fullbody.eval`` and related subprocesses.

The upstream layout places ``fullbody``, ``musclemimic``, and ``loco_mujoco``
under one directory (editable clone or ``site-packages``). Override with
``MYOSUITE_MUSCLEMIMIC_ROOT`` (path to any MuscleMimic project root).

Also builds commands for ``setup_demo_for_myo_fullbody`` when the active
MyoSuite interpreter does not import ``musclemimic``
(:func:`build_demo_cache_command`).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from myosuite.core.subprocess_orchestration import (
    build_python_command,
    prepend_env_path_var,
)


def _python_libdir(python_exe: Path) -> str | None:
    """Return the directory containing ``libpython`` for *python_exe*."""
    if not python_exe.is_file():
        return None
    try:
        proc = subprocess.run(
            [
                str(python_exe),
                "-c",
                (
                    "import sysconfig; "
                    "print(sysconfig.get_config_var('LIBDIR') or '')"
                ),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    raw = proc.stdout.strip().splitlines()
    lines = [ln.strip() for ln in raw if ln.strip()]
    if not lines:
        return None
    p = Path(lines[0])
    return str(p) if p.is_dir() else None


def _env_with_dyld_for_mjpython(
    base: dict[str, str], libdir: str | None
) -> dict[str, str]:
    """Prepend *libdir* to ``DYLD_LIBRARY_PATH`` for ``mjpython``."""
    return prepend_env_path_var(
        base_env=base,
        env_var="DYLD_LIBRARY_PATH",
        prefix_value=libdir,
    )


def resolve_musclemimic_root_from_env() -> Path | None:
    """Return ``MYOSUITE_MUSCLEMIMIC_ROOT`` if set and an existing directory.

    Returns:
        Resolved path, or ``None`` if the variable is unset.

    Raises:
        FileNotFoundError: If set but not a directory.
    """
    env_root = os.environ.get("MYOSUITE_MUSCLEMIMIC_ROOT")
    if not env_root:
        return None
    p = Path(env_root).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"MYOSUITE_MUSCLEMIMIC_ROOT is not a directory: {p}")
    return p


def resolve_musclemimic_exec_cwd() -> Path:
    """Return the working directory for ``python -m fullbody.eval``.

    Resolution:
        1. ``MYOSUITE_MUSCLEMIMIC_ROOT`` if set (must exist).
        2. Else import ``fullbody`` + ``musclemimic`` and use their common
           parent directory.

    Returns:
        Absolute directory to pass as ``cwd`` to subprocess.

    Raises:
        FileNotFoundError: If ``MYOSUITE_MUSCLEMIMIC_ROOT`` is set but invalid.
        ImportError: If MuscleMimic cannot be located.
    """
    from_env = resolve_musclemimic_root_from_env()
    if from_env is not None:
        return from_env

    err_fb: ImportError | None = None
    err_mm: ImportError | None = None
    try:
        import fullbody
    except ImportError as e:
        err_fb = e
    try:
        import musclemimic
    except ImportError as e:
        err_mm = e

    if err_fb is not None and err_mm is not None:
        raise ImportError(
            "full-body eval runs `python -m fullbody.eval`; this interpreter "
            "cannot import `fullbody` or `musclemimic`.\n\n"
            "Fix (pick one):\n"
            "  1) Point at a MuscleMimic checkout (no pip into this env):\n"
            "       export MYOSUITE_MUSCLEMIMIC_ROOT=/path/to/musclemimic\n"
            "     The directory must contain `fullbody/`. Create a venv there "
            "(`uv sync` or `pip install -e .`) so the subprocess can import "
            "packages.\n"
            "  2) Install upstream into a separate venv (Python>=3.11, "
            "**without** `myosuite[mjlab]` — warp pin conflict):\n"
            '       uv pip install "musclemimic @ git+https://github.com/'
            'amathislab/musclemimic.git"\n\n'
            f"Original errors: fullbody={err_fb!s}; musclemimic={err_mm!s}"
        ) from err_fb
    if err_fb is not None:
        raise ImportError(
            "`musclemimic` imports but `fullbody` does not. Use a full "
            "MuscleMimic tree (editable install or git package). "
            f"Original error: {err_fb}"
        ) from err_fb
    if err_mm is not None:
        raise ImportError(
            "`fullbody` imports but `musclemimic` does not. Reinstall the "
            f"full MuscleMimic repo. Original error: {err_mm}"
        ) from err_mm

    fb_file = getattr(fullbody, "__file__", None)
    mm_file = getattr(musclemimic, "__file__", None)
    if not fb_file or not mm_file:
        raise ImportError(
            "Could not resolve paths for `fullbody` / `musclemimic` "
            "(namespace layout?)."
        )
    fdir = Path(fb_file).resolve().parent
    mdir = Path(mm_file).resolve().parent
    if fdir.parent != mdir.parent:
        raise ImportError(
            "`fullbody` and `musclemimic` must share the same install root "
            f"(got {fdir.parent} vs {mdir.parent}). "
            "Set MYOSUITE_MUSCLEMIMIC_ROOT."
        )
    return fdir.parent


def _musclemimic_python_cmd(
    exec_cwd: Path,
    py_args: list[str],
) -> tuple[list[str], dict[str, str]]:
    """Build ``python`` + *py_args* for *exec_cwd*.

    Picks ``uv run``, ``.venv/bin/python``, or ``sys.executable``. Skips
    ``mjpython`` (no interactive MuJoCo window).
    """
    return build_python_command(
        exec_cwd=exec_cwd,
        py_args=py_args,
        override_env_var="MYOSUITE_MUSCLEMIMIC_PYTHON",
    )


def _demo_cache_snippet(env_name: str) -> str:
    """Build python snippet calling upstream demo-cache setup."""
    env = env_name.strip().lower()
    if env in ("myofullbody", "fullbody"):
        fn = "setup_demo_for_myo_fullbody"
    elif env in ("myobimanualarm", "bimanual"):
        fn = "setup_demo_for_bimanual"
    else:
        raise ValueError(
            f"Unsupported demo cache env_name={env_name!r}. "
            "Expected one of: MyoFullBody, MyoBimanualArm."
        )
    return f"from musclemimic.utils.demo_cache import {fn} as _f; _f()"


def build_demo_cache_command(
    exec_cwd: Path,
    env_name: str = "MyoFullBody",
) -> tuple[list[str], dict[str, str]]:
    """Build a command to run upstream demo-cache setup in *exec_cwd*.

    Uses the same interpreter selection as eval
    (``MYOSUITE_MUSCLEMIMIC_PYTHON``, ``uv run --directory``, or
    ``.venv/bin/python``) when the active env lacks ``musclemimic``.

    Args:
        exec_cwd: MuscleMimic project root (from
            :func:`resolve_musclemimic_exec_cwd`).

    Returns:
        ``argv`` and ``env`` for :func:`subprocess.run` with ``cwd=exec_cwd``.
    """
    snippet = _demo_cache_snippet(env_name)
    return _musclemimic_python_cmd(exec_cwd, ["-c", snippet])


def build_fullbody_eval_command(
    exec_cwd: Path,
    argv: list[str],
) -> tuple[list[str], dict[str, str]]:
    """Build ``python -m fullbody.eval`` (or ``uv run`` / ``mjpython``)."""
    override = os.environ.get("MYOSUITE_MUSCLEMIMIC_PYTHON")
    if override:
        return [override, "-m", "fullbody.eval", *argv], os.environ.copy()

    use_viewer = "--mujoco_viewer" in argv
    darwin = sys.platform == "darwin"

    venv_py = exec_cwd / ".venv" / "bin" / "python"
    mjpython_local = exec_cwd / ".venv" / "bin" / "mjpython"
    which_mj = shutil.which("mjpython")
    mjpython_exe: Path | None = None
    if mjpython_local.is_file():
        mjpython_exe = mjpython_local
    elif which_mj:
        cand = Path(which_mj).resolve()
        if cand.is_file():
            mjpython_exe = cand

    if darwin and use_viewer and mjpython_exe is not None:
        py_for_lib = venv_py if venv_py.is_file() else Path(sys.executable)
        libdir = _python_libdir(py_for_lib)
        env = _env_with_dyld_for_mjpython(os.environ.copy(), libdir)
        return [str(mjpython_exe), "-m", "fullbody.eval", *argv], env

    uv = shutil.which("uv")
    if uv and (exec_cwd / "pyproject.toml").is_file():
        return [
            uv,
            "run",
            "--directory",
            str(exec_cwd),
            "python",
            "-m",
            "fullbody.eval",
            *argv,
        ], os.environ.copy()

    if venv_py.is_file():
        return [str(venv_py), "-m", "fullbody.eval", *argv], os.environ.copy()

    return [sys.executable, "-m", "fullbody.eval", *argv], os.environ.copy()


__all__ = [
    "build_demo_cache_command",
    "build_fullbody_eval_command",
    "resolve_musclemimic_exec_cwd",
    "resolve_musclemimic_root_from_env",
]
