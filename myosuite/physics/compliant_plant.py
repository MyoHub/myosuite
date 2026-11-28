# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Runtime bridge for ``myo_sim >= 0.3.0`` series-elastic tendons and DGF muscles.

On ``myo_sim 0.2.5`` (``v0`` models), ``model.nuser_actuator == 0`` and every
helper below is a zero-overhead no-op, preserving bit-exact legacy ``mj_step``
and ``mujoco_warp`` behavior. On ``myo_sim >= 0.3.0`` (``v1`` models carrying
De Groote-Fregly user markers ``KIND_DGF = 1.0`` or ``KIND_DGF_RIGID = 2.0``),
these helpers size ``spec.nuserdata``, equilibrate fiber lengths on reset, and
dispatch CPU / GPU substeps through ``myo_sim.compliant`` / ``myo_sim.warp``.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

KIND_DGF_COMPLIANT = 1.0
KIND_DGF_RIGID = 2.0


def has_dgf_actuators(model: mujoco.MjModel) -> bool:
    """Return True if *model* carries ``myo_sim >= 0.3.0`` DGF actuator metadata."""
    if int(model.nu) == 0 or int(model.nuser_actuator) < 8:
        return False
    kinds = np.asarray(model.actuator_user[:, 0])
    return bool(np.any((kinds == KIND_DGF_COMPLIANT) | (kinds == KIND_DGF_RIGID)))


def has_compliant_tendons(model: mujoco.MjModel) -> bool:
    """Return True if *model* has explicit series-elastic fiber states (``KIND_DGF = 1``)."""
    if int(model.nu) == 0 or int(model.nuser_actuator) < 8:
        return False
    return bool(np.any(np.asarray(model.actuator_user[:, 0]) == KIND_DGF_COMPLIANT))


def detect_model_generation(model: mujoco.MjModel) -> str:
    """Classify *model* as ``'v1_compliant'``, ``'v1_rigid'``, or ``'v0'``."""
    if has_compliant_tendons(model):
        return "v1_compliant"
    if has_dgf_actuators(model):
        return "v1_rigid"
    return "v0"


def prepare_spec_for_plant(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Reserve ``userdata`` slots on *spec* when ``myo_sim.compliant`` is active."""
    try:
        from myo_sim import compliant  # type: ignore[import-not-found]
    except ImportError:
        return spec
    size_fn = getattr(compliant, "size_userdata", None)
    if callable(size_fn):
        size_fn(spec)
    return spec


class CpuCompliantPlant:
    """Lazy CPU ``myo_sim.compliant.CompliantTendons`` plant wrapper for one ``MjModel``."""

    def __init__(self, model: mujoco.MjModel) -> None:
        from myo_sim import compliant  # type: ignore[import-not-found]

        self._compliant = compliant
        self._model = model
        self._plant = compliant.CompliantTendons(model, install=True)
        try:
            from myo_sim.build.hand_data import place_tendon_nodes  # type: ignore[import-not-found]

            self._place_nodes: Any = place_tendon_nodes
        except ImportError:
            self._place_nodes = None

    @property
    def plant(self) -> Any:
        return self._plant

    def reset_and_forward(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Equilibrate series-elastic fiber lengths after ``reset`` / state writes."""
        if data.userdata is not None and data.userdata.size > 0:
            data.userdata[:] = 0.0
        if self._place_nodes is not None:
            self._place_nodes(model, data)
        self._compliant.forward(model, data, self._plant)

    def forward(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Recompute derived quantities with active DGF/compliant force callbacks."""
        self._compliant.forward(model, data, self._plant)

    def step(self, model: mujoco.MjModel, data: mujoco.MjData, nstep: int = 1) -> None:
        """Advance *nstep* substeps and refresh post-step derived quantities."""
        self._compliant.step(model, data, self._plant, int(nstep))
        self._compliant.forward(model, data, self._plant)


def get_or_attach_cpu_plant(
    owner: Any, model: mujoco.MjModel
) -> CpuCompliantPlant | None:
    """Return a cached :class:`CpuCompliantPlant` on *owner* if *model* uses DGF."""
    if not has_dgf_actuators(model):
        return None
    cached: tuple[mujoco.MjModel, CpuCompliantPlant] | None = getattr(
        owner, "_myosuite_compliant_plant", None
    )
    if cached is not None and cached[0] is model:
        return cached[1]
    plant = CpuCompliantPlant(model)
    owner._myosuite_compliant_plant = (model, plant)
    return plant


_MJLAB_HOOK_INSTALLED = False


def install_mjlab_compliant_hook() -> bool:
    """Route ``mjlab.sim.Simulation`` model creation through ``myo_sim.warp.put_model`` when needed.

    Idempotent: safe to call multiple times. ``v0`` models (``has_dgf_actuators == False``)
    use ``mjlab``'s stock ``mujoco_warp.put_model`` path unchanged.
    """
    global _MJLAB_HOOK_INSTALLED
    if _MJLAB_HOOK_INSTALLED:
        return True
    try:
        from mjlab.sim import Simulation  # type: ignore[import-not-found]
        import mujoco as _mj
        import warp as wp  # type: ignore[import-not-found]
        from myo_sim.warp import put_model as myo_warp_put_model  # type: ignore[import-not-found]
    except ImportError:
        return False

    orig_sim_init = Simulation.__init__
    orig_init_with_model = Simulation._init_with_model

    def _patched_sim_init(self: Any, *args: Any, **kwargs: Any) -> None:
        spec = kwargs.get("spec")
        if spec is not None:
            prepare_spec_for_plant(spec)
        orig_sim_init(self, *args, **kwargs)

    def _patched_init_with_model(self: Any, model: _mj.MjModel) -> None:
        if not has_dgf_actuators(model):
            orig_init_with_model(self, model)
            return
        self._mj_model = model
        self._mj_data = _mj.MjData(model)
        _mj.mj_forward(self._mj_model, self._mj_data)
        with wp.ScopedDevice(self.wp_device):
            wp_model, _ = myo_warp_put_model(self._mj_model)
            self._wp_model = wp_model
            self.cfg.apply_wp_opt(self._wp_model.opt)
            self._finish_init()

    Simulation.__init__ = _patched_sim_init  # type: ignore[method-assign]
    Simulation._init_with_model = _patched_init_with_model  # type: ignore[method-assign]
    _MJLAB_HOOK_INSTALLED = True
    return True
