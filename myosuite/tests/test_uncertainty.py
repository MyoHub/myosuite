# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Uncertainty bands: collapse without uncertainty, honest coverage, and the perturbed models."""

from __future__ import annotations

import copy
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.biomechanics.inverse_dynamics import clip_kinematics
from myosuite.biomechanics import uncertainty
from myosuite.biomechanics.static_optimization import (
    StaticOptimizer,
    StaticOptimizationConfig,
    static_optimization,
)
from myosuite.biomechanics.uncertainty import (
    UncertaintyConfig,
    UncertaintySolveError,
    perturbed_model,
    static_optimization_bands,
)

pytestmark = pytest.mark.tier1

_DT = 1 / 50
_CLIP = (1.0 + 0.6 * np.sin(2.5 * np.arange(0.0, 1.2, _DT)))[:, None]  # elbow flexion


@pytest.fixture(scope="module")
def elbow() -> mujoco.MjModel:
    env = make_env("myoElbowPose1D6MRandom-v0")
    yield env.unwrapped.model
    env.close()


def _force_at_full_activation(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[:], data.act[:] = qpos, 1.0
    mujoco.mj_forward(model, data)
    return data.actuator_force.copy()


def test_bands_collapse_without_uncertainty(elbow: mujoco.MjModel) -> None:
    bands = static_optimization_bands(
        elbow, _CLIP, _DT, config=UncertaintyConfig(samples=3)
    )
    for q in range(len(bands.quantiles)):
        np.testing.assert_allclose(
            bands.activation[q], bands.nominal.activation, atol=1e-9
        )
        np.testing.assert_allclose(
            bands.joint_torque[q], bands.nominal.joint_torque, atol=1e-9
        )


def test_bands_cover_the_truth(elbow: mujoco.MjModel) -> None:
    # Subjects with log-normal strength (CV 0.2, mean 1), drawn here independently of
    # the code under test: their minimum-effort activations must fall inside the
    # 90 % bands about (n - 1) * 0.9 / (n + 1) = 0.79 of the time for n = 15.
    sigma = np.sqrt(np.log1p(0.2**2))
    clip, hits, total = _CLIP[:30], 0, 0
    for trial in range(6):
        subject = copy.deepcopy(elbow)
        k = np.random.default_rng(100 + trial).lognormal(
            -0.5 * sigma**2, sigma, elbow.nu
        )
        subject.actuator_gainprm[:, 2] *= k
        subject.actuator_biasprm[:, 2] *= k
        kin = clip_kinematics(subject, clip, _DT)
        truth = static_optimization(subject, kin.qpos, kin.qvel, kin.qacc).activation
        config = UncertaintyConfig(samples=15, strength_cv=0.2, seed=trial)
        bands = static_optimization_bands(elbow, clip, _DT, config=config)
        lo, hi = bands.activation[0], bands.activation[-1]
        active = truth > 0.01
        hits += int(((truth >= lo - 1e-9) & (truth <= hi + 1e-9))[active].sum())
        total += int(active.sum())
    assert 0.6 < hits / total < 0.95


def test_wider_priors_give_wider_bands(elbow: mujoco.MjModel) -> None:
    def width(cv: float) -> float:
        b = static_optimization_bands(
            elbow, _CLIP, _DT, config=UncertaintyConfig(samples=15, strength_cv=cv)
        )
        return float((b.activation[-1] - b.activation[0]).mean())

    assert width(0.3) > 1.5 * width(0.1) > 0.0


def test_joint_noise_and_masses_widen_the_torque_bands(elbow: mujoco.MjModel) -> None:
    noise = static_optimization_bands(
        elbow,
        _CLIP,
        _DT,
        lowpass_hz=6.0,
        config=UncertaintyConfig(samples=15, joint_noise_deg=0.5),
    )
    mass = static_optimization_bands(
        elbow, _CLIP, _DT, config=UncertaintyConfig(samples=15, mass_cv=0.1)
    )
    for bands in (noise, mass):
        assert (bands.joint_torque[-1] - bands.joint_torque[0]).mean() > 0.05  # N m
    # Strength does not change the motion's torques.
    strength = static_optimization_bands(
        elbow, _CLIP, _DT, config=UncertaintyConfig(samples=5, strength_cv=0.3)
    )
    np.testing.assert_allclose(
        strength.joint_torque[0], strength.joint_torque[-1], atol=1e-9
    )


def test_perturbed_model_scales_strength_and_mass_and_keeps_the_original() -> None:
    xml = """
    <mujoco><worldbody><site name="o" pos="0 0 .1"/>
      <body><joint name="j" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 .3 0 0" size=".02" mass="1"/>
        <site name="i" pos=".2 0 .05"/></body></worldbody>
      <tendon><spatial name="t"><site site="o"/><site site="i"/></spatial></tendon>
      <actuator><muscle name="m" tendon="t" force="-1" scale="200"/></actuator></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    original = copy.deepcopy(model)
    q = np.array([0.3])
    # Mass alone leaves the automatic peak force where it was, though acc0 changes.
    heavy = perturbed_model(
        model, np.random.default_rng(0), UncertaintyConfig(mass_cv=0.5)
    )
    assert heavy.body_mass[1] != pytest.approx(model.body_mass[1])
    np.testing.assert_allclose(
        heavy.body_inertia[1] / model.body_inertia[1],
        heavy.body_mass[1] / model.body_mass[1],
    )
    np.testing.assert_allclose(
        _force_at_full_activation(heavy, q),
        _force_at_full_activation(model, q),
        rtol=1e-12,
    )
    strong = perturbed_model(
        model, np.random.default_rng(0), UncertaintyConfig(strength_cv=0.5)
    )
    ratio = _force_at_full_activation(strong, q) / _force_at_full_activation(model, q)
    assert abs(ratio[0] - 1) > 0.01
    np.testing.assert_array_equal(model.body_mass, original.body_mass)
    np.testing.assert_array_equal(model.actuator_gainprm, original.actuator_gainprm)


def test_strength_scales_active_and_passive_force_alike(elbow: mujoco.MjModel) -> None:
    strong = perturbed_model(
        elbow, np.random.default_rng(1), UncertaintyConfig(strength_cv=0.3)
    )
    q = np.array([0.1])  # extensors stretched: passive force present
    active = (_force_at_full_activation(strong, q) - _force_at(strong, q, 0.0)) / (
        _force_at_full_activation(elbow, q) - _force_at(elbow, q, 0.0)
    )
    stretched = np.abs(_force_at(elbow, q, 0.0)) > 1.0
    assert stretched.any()
    np.testing.assert_allclose(
        _force_at(strong, q, 0.0)[stretched] / _force_at(elbow, q, 0.0)[stretched],
        active[stretched],
    )
    assert np.ptp(active) > 0.1  # one factor per muscle


def _force_at(model: mujoco.MjModel, qpos: np.ndarray, level: float) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[:], data.act[:] = qpos, level
    mujoco.mj_forward(model, data)
    return data.actuator_force.copy()


def test_bands_are_reproducible(elbow: mujoco.MjModel) -> None:
    config = UncertaintyConfig(samples=4, strength_cv=0.2, seed=3)
    a = static_optimization_bands(elbow, _CLIP, _DT, config=config)
    b = static_optimization_bands(elbow, _CLIP, _DT, config=config)
    np.testing.assert_array_equal(a.activation, b.activation)


def test_samples_keep_the_nominal_weights(
    elbow: mujoco.MjModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Weaker sampled muscles must not rescale the reserves: only the samples vary."""
    seen = []

    def spy(*args, config=None, **kwargs):
        seen.append(config)
        return static_optimization(*args, config=config, **kwargs)

    owned = StaticOptimizer._from_owned_model

    def owned_spy(cls, model, config=None):
        seen.append(config)
        return owned(model, config)

    monkeypatch.setattr(uncertainty, "static_optimization", spy)
    monkeypatch.setattr(StaticOptimizer, "_from_owned_model", classmethod(owned_spy))
    static_optimization_bands(
        elbow, _CLIP, _DT, config=UncertaintyConfig(samples=3, strength_cv=0.5)
    )
    nominal = StaticOptimizer(elbow).config
    assert len(seen) == 4 and all(c == nominal for c in seen)


def test_nominal_iteration_failure_stops_before_sampling(
    elbow: mujoco.MjModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_sampling(*args, **kwargs):
        pytest.fail("a failed nominal solve must stop before sampling")

    monkeypatch.setattr(uncertainty, "perturbed_model", no_sampling)
    with pytest.raises(
        UncertaintySolveError, match="nominal.*maximum iterations"
    ) as error:
        static_optimization_bands(
            elbow, _CLIP[:8], _DT, optimization=StaticOptimizationConfig(max_iter=1)
        )
    assert error.value.sample_index is None
    assert len(error.value.frame_indices) > 0
    assert (error.value.result.iterations == 1).all()
    assert (error.value.result.primal_residual > 0).any()


def test_sample_iteration_failure_is_reported(
    elbow: mujoco.MjModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    solve = StaticOptimizer.solve

    def limited(self, *args, **kwargs):
        if not np.array_equal(self.model.actuator_gainprm, elbow.actuator_gainprm):
            self.config = replace(self.config, max_iter=1)
        return solve(self, *args, **kwargs)

    monkeypatch.setattr(StaticOptimizer, "solve", limited)
    with pytest.raises(
        UncertaintySolveError, match="sample 0.*maximum iterations"
    ) as error:
        static_optimization_bands(
            elbow, _CLIP[:8], _DT, config=UncertaintyConfig(samples=3, strength_cv=0.2)
        )
    assert error.value.sample_index == 0
    assert (error.value.result.iterations == 1).all()


@pytest.mark.parametrize("nominal", [False, True])
def test_inaccurate_status_needs_explicit_opt_in_and_keeps_diagnostics(
    elbow: mujoco.MjModel, monkeypatch: pytest.MonkeyPatch, nominal: bool
) -> None:
    config = UncertaintyConfig(samples=2, strength_cv=0.2, seed=4)
    clean = static_optimization_bands(elbow, _CLIP[:8], _DT, config=config)
    solve = StaticOptimizer.solve

    def inaccurate(self, *args, **kwargs):
        result = solve(self, *args, **kwargs)
        is_nominal = np.array_equal(self.model.actuator_gainprm, elbow.actuator_gainprm)
        if is_nominal == nominal:
            result = replace(result, status=np.asarray("solved inaccurate"))
        return result

    monkeypatch.setattr(StaticOptimizer, "solve", inaccurate)
    with pytest.raises(UncertaintySolveError, match="solved inaccurate"):
        static_optimization_bands(elbow, _CLIP[:8], _DT, config=config)
    bands = static_optimization_bands(
        elbow, _CLIP[:8], _DT, config=replace(config, allow_inaccurate=True)
    )
    np.testing.assert_array_equal(bands.activation, clean.activation)
    np.testing.assert_array_equal(bands.sample_iterations, clean.sample_iterations)
    np.testing.assert_array_equal(
        bands.sample_primal_residual, clean.sample_primal_residual
    )
    np.testing.assert_array_equal(
        bands.sample_dual_residual, clean.sample_dual_residual
    )
    assert bands.sample_status.shape == (2, 8)
    if nominal:
        assert bands.nominal.status[0] == "solved inaccurate"
    else:
        assert (bands.sample_status[:, 0] == "solved inaccurate").all()


def test_nonfinite_sample_is_rejected_even_with_inaccurate_opt_in(
    elbow: mujoco.MjModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    solve = StaticOptimizer.solve
    frames = []

    def nonfinite(self, *args, **kwargs):
        result = solve(self, *args, **kwargs)
        if not np.array_equal(self.model.actuator_gainprm, elbow.actuator_gainprm):
            frames.append(result)
        if len(frames) == 2:
            activation = result.activation.copy()
            activation[0] = np.nan
            result = replace(result, activation=activation)
        return result

    monkeypatch.setattr(StaticOptimizer, "solve", nonfinite)
    with pytest.raises(UncertaintySolveError, match="nonfinite") as error:
        static_optimization_bands(
            elbow,
            _CLIP[:8],
            _DT,
            config=UncertaintyConfig(samples=2, strength_cv=0.2, allow_inaccurate=True),
        )
    np.testing.assert_array_equal(error.value.frame_indices, [1])


def test_owned_samples_match_copying_path_and_preserve_input(
    elbow: mujoco.MjModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An extra model copy must change neither the sampled prior nor the outputs."""
    original = copy.deepcopy(elbow)
    config = UncertaintyConfig(
        samples=3, strength_cv=0.2, mass_cv=0.1, joint_noise_deg=0.1, seed=3
    )

    def copying(cls, model, config=None):
        return cls(model, config)

    with monkeypatch.context() as patch:
        patch.setattr(StaticOptimizer, "_from_owned_model", classmethod(copying))
        reference = static_optimization_bands(elbow, _CLIP[:20], _DT, config=config)
    actual = static_optimization_bands(elbow, _CLIP[:20], _DT, config=config)
    for name in (
        "activation",
        "actuator_force",
        "reserve",
        "joint_torque",
        "contact_force",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(reference, name))
    for name in (
        "body_mass",
        "body_inertia",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_actearly",
        "geom_margin",
    ):
        np.testing.assert_array_equal(getattr(elbow, name), getattr(original, name))
    assert elbow.opt.jacobian == original.opt.jacobian
    assert elbow.opt.disableflags == original.opt.disableflags
