# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Runtime compatibility helpers for DreamerV3 on modern JAX and NumPy."""

from __future__ import annotations

import contextlib
import functools
import importlib
import os
from pathlib import Path
import sys
import types
from typing import Any
import warnings


def _import_real_package(pkg_name: str, marker_subpath: str) -> Any:
    """Import ``pkg_name`` while skipping outer repo-checkout directories on ``sys.path``."""
    mod = importlib.import_module(pkg_name)
    mod_file = getattr(mod, "__file__", None)
    if mod_file is not None:
        return mod
    mod_paths = [Path(p).resolve() for p in getattr(mod, "__path__", [])]
    if any((p / marker_subpath).exists() for p in mod_paths):
        return mod

    bad_parents = {str(p.parent) for p in mod_paths}
    saved_path = sys.path[:]
    try:
        sys.path = [
            p for p in sys.path if str(Path(p or ".").resolve()) not in bad_parents
        ]
        for key in [
            k for k in sys.modules if k == pkg_name or k.startswith(f"{pkg_name}.")
        ]:
            del sys.modules[key]
        return importlib.import_module(pkg_name)
    finally:
        sys.path = saved_path


def import_myosuite() -> Any:
    """Import ``myosuite`` while avoiding PEP 420 namespace shadowing on ``sys.path``.

    Returns:
        Imported ``myosuite`` module.
    """
    return _import_real_package("myosuite", "version.py")


def import_dreamerv3() -> tuple[Any, Path]:
    """Import ``dreamerv3`` (and ``embodied``) while resolving ``configs.yaml``.

    Returns:
        Tuple of ``(dreamerv3_module, configs_yaml_path)``.
    """
    try:
        dv3 = _import_real_package("dreamerv3", "configs.yaml")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import 'dreamerv3'. Clone https://github.com/danijar/dreamerv3 "
            "and add it to PYTHONPATH (or install a wheel-enabled fork)."
        ) from exc

    mod_file = getattr(dv3, "__file__", None)
    if mod_file is not None:
        cfg_path = Path(mod_file).with_name("configs.yaml")
        if cfg_path.is_file():
            return dv3, cfg_path
    for p in getattr(dv3, "__path__", []):
        candidate = Path(p) / "configs.yaml"
        if candidate.is_file():
            _import_real_package("embodied", "core")
            return dv3, candidate
    raise FileNotFoundError(
        "Could not locate dreamerv3/configs.yaml in imported package."
    )


def resolve_jax_platform(platform: str) -> str:
    """Return ``platform`` if available in JAX, falling back cleanly to ``'cpu'``."""
    if platform != "cuda":
        return platform
    try:
        from jax._src import xla_bridge

        backends = xla_bridge.backends()
        if "cuda" in backends or "gpu" in backends:
            return "cuda"
    except Exception:
        return "cuda"
    warnings.warn(
        "CUDA JAX backend is not installed; falling back from platform='cuda' to 'cpu'.",
        RuntimeWarning,
        stacklevel=2,
    )
    return "cpu"


@functools.cache
def apply_dreamer_jax_compat() -> None:
    """Apply JAX >= 0.8, Optax, and XLA flag shims so ``danijar/dreamerv3`` runs cleanly."""
    import jax

    orig_jit = jax.jit
    try:
        orig_jit(lambda x: x, None, None)
        needs_jit_shim = False
    except TypeError:
        needs_jit_shim = True

    if needs_jit_shim:
        pos_names = (
            "in_shardings",
            "out_shardings",
            "static_argnums",
            "static_argnames",
            "donate_argnums",
        )

        @functools.wraps(orig_jit)
        def _positional_compat_jit(fun: Any, *args: Any, **kwargs: Any) -> Any:
            if not args:
                return orig_jit(fun, **kwargs)
            mapped = dict(kwargs)
            for name, val in zip(pos_names, args):
                if val is not None or name in ("in_shardings", "out_shardings"):
                    mapped.setdefault(name, val)
            return orig_jit(fun, **mapped)

        jax.jit = _positional_compat_jit  # type: ignore[assignment]

    if hasattr(jax, "shard_map") and "jax.experimental.shard_map" not in sys.modules:
        try:
            import jax.experimental.shard_map  # noqa: F401
        except ModuleNotFoundError:
            mod = types.ModuleType("jax.experimental.shard_map")

            def _shard_map(
                f: Any,
                mesh: Any,
                in_specs: Any,
                out_specs: Any,
                check_rep: bool = False,
                **kw: Any,
            ) -> Any:
                return jax.shard_map(
                    f,
                    mesh=mesh,
                    in_specs=in_specs,
                    out_specs=out_specs,
                    check_vma=check_rep,
                    **kw,
                )

            mod.shard_map = _shard_map  # type: ignore[attr-defined]
            sys.modules["jax.experimental.shard_map"] = mod
            if hasattr(jax, "experimental"):
                jax.experimental.shard_map = mod  # type: ignore[attr-defined]

    src_cfg = getattr(getattr(jax, "_src", None), "config", None)
    if src_cfg is not None:
        if not hasattr(src_cfg, "explicit_device_get_scope"):
            src_cfg.explicit_device_get_scope = lambda: jax.transfer_guard("allow")
        if not hasattr(src_cfg, "explicit_device_put_scope"):
            src_cfg.explicit_device_put_scope = lambda: jax.transfer_guard("allow")

    if hasattr(jax, "set_mesh"):
        try:
            mesh_cls = jax.sharding.Mesh
            stack_attr = "_dreamerv3_set_mesh_stack"

            def _mesh_enter(self: Any) -> Any:
                cm = jax.set_mesh(self)
                stack = getattr(self, stack_attr, None)
                if stack is None:
                    stack = contextlib.ExitStack()
                    object.__setattr__(self, stack_attr, stack)
                stack.enter_context(cm)
                return self

            def _mesh_exit(self: Any, exc_type: Any, exc: Any, tb: Any) -> bool:
                stack = getattr(self, stack_attr, None)
                if stack is not None:
                    return bool(stack.__exit__(exc_type, exc, tb))
                return False

            mesh_cls.__enter__ = _mesh_enter  # type: ignore[method-assign]
            mesh_cls.__exit__ = _mesh_exit  # type: ignore[method-assign]
        except Exception:
            pass

    try:
        import optax

        if hasattr(optax, "tree") and hasattr(optax.tree, "norm"):
            optax.global_norm = optax.tree.norm  # type: ignore[assignment]
    except Exception:
        pass

    import_dreamerv3()

    if os.name == "nt":
        try:
            import elements.path as epath

            orig_path_init = epath.Path.__init__

            @functools.wraps(orig_path_init)
            def _posix_path_init(self: Any, path: Any) -> None:
                orig_path_init(self, str(path).replace("\\", "/"))

            epath.Path.__init__ = _posix_path_init  # type: ignore[method-assign]
        except Exception:
            pass

    jax_ver = tuple(int(x) for x in jax.__version__.split(".")[:3] if x.isdigit())
    if jax_ver >= (0, 4, 35):
        try:
            from embodied.jax import internal as jax_internal

            orig_setup = jax_internal.setup

            @functools.wraps(orig_setup)
            def _compat_setup(*args: Any, **kwargs: Any) -> Any:
                res = orig_setup(*args, **kwargs)
                flags = os.environ.get("XLA_FLAGS", "")
                if flags:
                    cleaned = [
                        tok
                        for tok in flags.split()
                        if not tok.startswith(
                            (
                                "--xla_gpu_enable_triton_softmax_fusion",
                                "--xla_gpu_graph_level",
                            )
                        )
                    ]
                    os.environ["XLA_FLAGS"] = " ".join(cleaned)
                return res

            jax_internal.setup = _compat_setup
        except ImportError:
            pass
