# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Polynomial joint couplings (MuJoCo ``mjEQ_JOINT`` equalities) on any array backend.

A joint equality ties a dependent joint ``y`` to a master joint ``x`` through a
quartic in their offsets from the reference positions ``qpos0``::

    y - y0 = c0 + c1 (x - x0) + c2 (x - x0)^2 + c3 (x - x0)^3 + c4 (x - x0)^4

or locks ``y - y0 = c0`` when there is no master. The knee mechanism (tibia
translations, patella) and the shoulder rhythm of the MyoSuite models are such
couplings. MuJoCo enforces them softly: a state written directly (``qpos0``, a
keyframe, a clip frame, reset noise) that is off the polynomials is pulled back
over the next steps, and until then ``actuator_moment`` describes a pose the
coupled joints cannot reach.

:func:`project_joint_couplings` puts every dependent joint on its polynomial, in
place, for NumPy arrays (CPU) or batched Torch tensors (mjlab).
:func:`projected_joint_couplings` returns the projected state as new arrays instead,
so Torch autograd can differentiate through it (e.g. forward kinematics of a
learned model's independent joints).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np


@dataclasses.dataclass(frozen=True, eq=False)
class JointCouplings:
    """The active joint equalities of a model, as address and coefficient arrays.

    Rows are ordered so that a coupling whose master is itself coupled comes after
    the coupling that writes that master (``level_ends`` closes each pass).

    Attributes:
        locked_qpos: ``(l,)`` qpos addresses of the joints locked without a master.
        locked_dof: ``(l,)`` their dof addresses.
        locked_value: ``(l,)`` their positions, ``qpos0 + c0``.
        dep_qpos: ``(k,)`` qpos addresses of the dependent joints.
        dep_dof: ``(k,)`` their dof addresses.
        dep_ref: ``(k,)`` their reference positions (``qpos0``).
        master_qpos: ``(k,)`` qpos addresses of the master joints.
        master_dof: ``(k,)`` their dof addresses.
        master_ref: ``(k,)`` their reference positions.
        coef: ``(k, 5)`` polynomial coefficients ``c0 .. c4``.
        level_ends: End row of each pass (a single pass without chained couplings).
    """

    locked_qpos: Any
    locked_dof: Any
    locked_value: Any
    dep_qpos: Any
    dep_dof: Any
    dep_ref: Any
    master_qpos: Any
    master_dof: Any
    master_ref: Any
    coef: Any
    level_ends: tuple[int, ...]

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> JointCouplings:
        """Collect the joint equalities of *model* that are active at reset (``eq_active0``).

        Args:
            model: Compiled model.

        Returns:
            The couplings, as NumPy arrays.

        Raises:
            ValueError: If a joint is the dependent of two equalities, or the
                equalities form a cycle.
        """
        eq = np.flatnonzero(
            (model.eq_type == mujoco.mjtEq.mjEQ_JOINT) & (model.eq_active0 != 0)
        )
        dep, master = model.eq_obj1id[eq], model.eq_obj2id[eq]
        if len(np.unique(dep)) != len(dep):
            raise ValueError("A joint is the dependent of more than one joint equality")
        locked, coupled = eq[master < 0], eq[master >= 0]
        dep_c, master_c = model.eq_obj1id[coupled], model.eq_obj2id[coupled]
        # Pass of each coupling: one after the pass that writes its master.
        row_of = {int(j): r for r, j in enumerate(dep_c)}
        level = np.zeros(len(coupled), dtype=int)
        for _ in range(len(coupled) + 1):
            new = np.array(
                [level[row_of[m]] + 1 if m in row_of else 0 for m in master_c.tolist()],
                dtype=int,
            )
            if np.array_equal(new, level):
                break
            level = new
        else:
            raise ValueError("Joint equalities form a cycle")
        order = np.argsort(level, kind="stable")
        coupled, dep_c, master_c = coupled[order], dep_c[order], master_c[order]
        level_ends = tuple(int(e) for e in np.cumsum(np.bincount(level[order])))
        dep_l = model.eq_obj1id[locked]
        qadr, dadr = model.jnt_qposadr, model.jnt_dofadr
        return cls(
            locked_qpos=qadr[dep_l].astype(np.int64),
            locked_dof=dadr[dep_l].astype(np.int64),
            locked_value=model.qpos0[qadr[dep_l]] + model.eq_data[locked, 0],
            dep_qpos=qadr[dep_c].astype(np.int64),
            dep_dof=dadr[dep_c].astype(np.int64),
            dep_ref=model.qpos0[qadr[dep_c]].copy(),
            master_qpos=qadr[master_c].astype(np.int64),
            master_dof=dadr[master_c].astype(np.int64),
            master_ref=model.qpos0[qadr[master_c]].copy(),
            coef=model.eq_data[coupled, :5].copy(),
            level_ends=level_ends,
        )

    def map_arrays(self, fn: Callable[[np.ndarray], Any]) -> JointCouplings:
        """These couplings with every array passed through *fn*.

        Args:
            fn: Array conversion, e.g. to Torch tensors on the simulation device.

        Returns:
            The converted couplings.
        """
        return dataclasses.replace(
            self,
            **{
                f.name: fn(getattr(self, f.name))
                for f in dataclasses.fields(self)
                if f.name != "level_ends"
            },
        )

    def as_torch(self, device: Any = None, dtype: Any = None) -> JointCouplings:
        """These couplings as Torch tensors: addresses as ``int64``, values as *dtype*.

        Args:
            device: Torch device of the tensors.
            dtype: Floating dtype of the reference positions and coefficients
                (default: Torch's default dtype), that of the states to project.

        Returns:
            The converted couplings.
        """
        import torch  # noqa: PLC0415

        values = dtype or torch.get_default_dtype()

        def convert(a: np.ndarray) -> Any:
            kind = values if a.dtype.kind == "f" else torch.int64
            return torch.as_tensor(a, dtype=kind, device=device)

        return self.map_arrays(convert)


def _polynomial(coef: Any, x: Any) -> tuple[Any, Any]:
    """``p(x)`` and ``p'(x)`` of the quartics with coefficients ``coef`` (Horner's scheme)."""
    value, slope = coef[:, 4], 4 * coef[:, 4]
    for i in (3, 2, 1):
        value = value * x + coef[:, i]
        slope = slope * x + i * coef[:, i]
    return value * x + coef[:, 0], slope


def project_joint_couplings(
    couplings: JointCouplings, qpos: Any, qvel: Any | None = None
) -> None:
    """Write every coupled joint from its master, in place.

    A dependent position becomes ``y0 + p(x - x0)`` and, with *qvel*, its
    velocity ``p'(x - x0) * xdot``, the velocity along the coupling. Locked
    joints get ``y0 + c0`` and zero velocity. Masters are not moved.

    Args:
        couplings: Couplings whose arrays are of the backend (and device) of *qpos*.
        qpos: Positions ``(..., nq)``, a NumPy array or Torch tensor.
        qvel: Optional velocities ``(..., nv)`` of the same backend.
    """
    c = couplings
    qpos[..., c.locked_qpos] = c.locked_value
    if qvel is not None:
        qvel[..., c.locked_dof] = 0.0
    start = 0
    for end in c.level_ends:
        x = qpos[..., c.master_qpos[start:end]] - c.master_ref[start:end]
        value, slope = _polynomial(c.coef[start:end], x)
        qpos[..., c.dep_qpos[start:end]] = c.dep_ref[start:end] + value
        if qvel is not None:
            qvel[..., c.dep_dof[start:end]] = slope * qvel[..., c.master_dof[start:end]]
        start = end


def _with_columns(x: Any, index: Any, values: Any) -> Any:
    """A copy of *x* with ``x[..., index] = values`` (out of place, differentiable in Torch)."""
    if isinstance(x, np.ndarray):
        out = x.copy()
        out[..., index] = values
        return out
    values = values.expand(*x.shape[:-1], index.shape[0])
    return x.index_copy(-1, index, values)


def projected_joint_couplings(
    couplings: JointCouplings, qpos: Any, qvel: Any | None = None
) -> tuple[Any, Any | None]:
    """The state with every coupled joint written from its master, as new arrays.

    The same projection as :func:`project_joint_couplings`, pass by pass in the
    couplings' topological order, but without writing into *qpos* or *qvel*: with
    Torch tensors the result is differentiable in the masters (a dependent's input
    value gets a zero gradient, it is overwritten).

    Args:
        couplings: Couplings whose arrays are of the backend, device and (floating)
            dtype of *qpos* (see :meth:`JointCouplings.as_torch`).
        qpos: Positions ``(..., nq)``, a NumPy array or Torch tensor.
        qvel: Optional velocities ``(..., nv)`` of the same backend.

    Returns:
        ``(qpos, qvel)`` projected; ``qvel`` is ``None`` if none was given.
    """
    c = couplings
    if len(c.locked_qpos):
        qpos = _with_columns(qpos, c.locked_qpos, c.locked_value)
        if qvel is not None:
            qvel = _with_columns(qvel, c.locked_dof, 0 * c.locked_value)
    start = 0
    for end in c.level_ends:
        x = qpos[..., c.master_qpos[start:end]] - c.master_ref[start:end]
        value, slope = _polynomial(c.coef[start:end], x)
        if qvel is not None:
            qvel = _with_columns(
                qvel, c.dep_dof[start:end], slope * qvel[..., c.master_dof[start:end]]
            )
        qpos = _with_columns(qpos, c.dep_qpos[start:end], c.dep_ref[start:end] + value)
        start = end
    return qpos, qvel
