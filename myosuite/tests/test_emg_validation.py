# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Unit and mutation-sensitivity tests for ``myosuite.biomechanics.emg_validation``."""

from __future__ import annotations

import copy
from pathlib import Path

import mujoco
import numpy as np
import pytest

import myosuite.biomechanics.emg_validation as emg_validation_mod
from myosuite.biomechanics.emg_validation import (
    ChannelBinding,
    EMGChannelMap,
    EMGTrial,
    actuator_pose_sensitivity,
    compare_emg_validation_reports,
    musculoskeletal_fingerprint,
    spatial_site_channel_map,
    summarize_emg_validation,
    validate_trial,
)
from myosuite.biomechanics.uncertainty import UncertaintyConfig
from myosuite.integrations.emg import (
    EMG_TRIAL_META_KEY,
    NpzDirectoryEMGSource,
    load_emg_trial,
    save_emg_trial,
)
from myosuite.integrations.mm_evalkit import (
    compare_emg_reports_mm_evalkit,
    export_mm_evalkit_rollout,
    write_mm_evalkit_policy_config,
)
from myosuite.tests.support.optional_deps import needs_myo_sim_fingerprint

pytestmark = pytest.mark.tier1

_TWO_MUSCLE_ARM_XML = """
<mujoco model="two_muscle_hinge">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="upper" pos="0 0 0.5">
      <site name="flex_orig" pos="0.04 0 0.12"/>
      <site name="ext_orig" pos="-0.04 0 0.12"/>
      <site name="lig_orig" pos="0.02 0 0.04"/>
      <site name="elec_flex" pos="0.04 0 0.06"/>
      <site name="elec_ext" pos="-0.04 0 0.06"/>
      <body name="forearm" pos="0 0 0">
        <inertial pos="0 0 -0.15" mass="1.2" diaginertia="0.012 0.012 0.002"/>
        <joint name="elbow_flexion" type="hinge" axis="0 -1 0" range="-1.5 1.5" damping="0.8"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.30" size="0.025"/>
        <site name="flex_ins" pos="0.035 0 -0.05"/>
        <site name="ext_ins" pos="-0.035 0 -0.05"/>
        <site name="lig_ins" pos="0.02 0 -0.04"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <spatial name="bic_tendon"><site site="flex_orig"/><site site="flex_ins"/></spatial>
    <spatial name="tri_tendon"><site site="ext_orig"/><site site="ext_ins"/></spatial>
    <spatial name="capsule_ligament" stiffness="25.0" damping="0.5" springlength="0.075 0.095">
      <site site="lig_orig"/><site site="lig_ins"/>
    </spatial>
  </tendon>
  <actuator>
    <muscle name="biceps_r" tendon="bic_tendon" force="120" lengthrange="0.09 0.20"/>
    <muscle name="triceps_r" tendon="tri_tendon" force="120" lengthrange="0.09 0.20"/>
  </actuator>
</mujoco>
"""


def test_channel_map_cross_version_resolution_and_npz_roundtrip(tmp_path: Path) -> None:
    model = mujoco.MjModel.from_xml_string(_TWO_MUSCLE_ARM_XML)
    spatial_map = spatial_site_channel_map(
        model, ("elec_flex", "elec_ext"), sigma_m=0.03, top_k=1
    )
    res_spatial = spatial_map.resolve(model, ("elec_flex", "elec_ext"))
    assert res_spatial.channels == ("elec_flex", "elec_ext")
    assert res_spatial.matrix[0, 0] == pytest.approx(1.0)
    assert res_spatial.matrix[1, 1] == pytest.approx(1.0)
    assert EMGChannelMap.from_dict(spatial_map.to_dict()) == spatial_map

    subcomp_xml = """
    <mujoco>
      <worldbody>
        <body name="arm">
          <joint name="gh" type="hinge"/>
          <geom type="sphere" size="0.05"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="DELT1" joint="gh" gear="1"/>
        <motor name="INFSP1_r" joint="gh" gear="1"/>
        <motor name="INFSP2_r" joint="gh" gear="1"/>
        <motor name="FCR_r" joint="gh" gear="1"/>
        <motor name="FCU_r" joint="gh" gear="1"/>
      </actuator>
    </mujoco>
    """
    m_sub = mujoco.MjModel.from_xml_string(subcomp_xml)
    cmap = EMGChannelMap(
        name="cross_pr_map",
        bindings=(
            ChannelBinding(channel="delt", actuators=("DELT1_r",)),
            ChannelBinding(channel="infsp", actuators=("INFSP_r",)),
            ChannelBinding(channel="flex", fmg="Wrist-Flexor", side="r"),
        ),
    )
    res = cmap.resolve(m_sub, ("delt", "infsp", "flex"))
    assert res.channels == ("delt", "infsp", "flex")
    assert np.allclose(res.matrix[0], [1.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(res.matrix[1], [0.0, 0.5, 0.5, 0.0, 0.0])
    assert np.allclose(res.matrix[2], [0.0, 0.0, 0.0, 0.5, 0.5])

    trial = EMGTrial(
        trial_id="unit_clip",
        time=np.linspace(0.0, 0.5, 51),
        signals=np.column_stack([np.linspace(0.0, 1.0, 51), np.linspace(1.0, 0.0, 51)]),
        channel_names=("elec_flex", "elec_ext"),
        signal_kind="mvc_normalized",
        qpos=np.linspace(0.1, 0.8, 51)[:, None],
        meta={"subject": "S01"},
    )
    npz_path = tmp_path / "unit_clip.npz"
    save_emg_trial(npz_path, trial)
    loaded = load_emg_trial(npz_path)
    assert loaded.trial_id == "unit_clip"
    assert loaded.channel_names == ("elec_flex", "elec_ext")
    assert np.allclose(loaded.signals, trial.signals)
    assert (
        loaded.qpos is not None
        and trial.qpos is not None
        and np.allclose(loaded.qpos, trial.qpos)
    )
    assert loaded.meta["subject"] == "S01"

    assert emg_validation_mod.EMG_TRIAL_META_KEY is EMG_TRIAL_META_KEY
    assert emg_validation_mod.save_emg_trial is save_emg_trial
    assert emg_validation_mod.load_emg_trial is load_emg_trial
    assert emg_validation_mod.export_mm_evalkit_rollout is export_mm_evalkit_rollout
    assert (
        emg_validation_mod.write_mm_evalkit_policy_config
        is write_mm_evalkit_policy_config
    )

    dir_src = NpzDirectoryEMGSource(tmp_path)
    assert dir_src.trial_ids() == ("unit_clip",)
    assert dir_src.get_trial("unit_clip").trial_id == "unit_clip"


@needs_myo_sim_fingerprint
def test_musculoskeletal_fingerprint_detects_ligament_and_muscle_mutations() -> None:
    m_base = mujoco.MjModel.from_xml_string(_TWO_MUSCLE_ARM_XML)
    m_same = mujoco.MjModel.from_xml_string(_TWO_MUSCLE_ARM_XML)
    fp_base = musculoskeletal_fingerprint(m_base)
    assert fp_base == musculoskeletal_fingerprint(m_same)

    m_lig = copy.deepcopy(m_base)
    m_lig.tendon_stiffness[m_lig.tendon("capsule_ligament").id] = 60.0
    assert musculoskeletal_fingerprint(m_lig) != fp_base

    m_force = copy.deepcopy(m_base)
    m_force.actuator_gainprm[0, 2] *= 1.25
    assert musculoskeletal_fingerprint(m_force) != fp_base

    m_geom = mujoco.MjModel.from_xml_string(
        _TWO_MUSCLE_ARM_XML.replace('pos="0.035 0 -0.05"', 'pos="0.055 0 -0.05"')
    )
    assert musculoskeletal_fingerprint(m_geom) != fp_base


@needs_myo_sim_fingerprint
def test_inverse_emg_feasibility_and_known_bad_mutations() -> None:
    model = mujoco.MjModel.from_xml_string(_TWO_MUSCLE_ARM_XML)
    data = mujoco.MjData(model)
    dt_sim = model.opt.timestep
    n_steps = 800
    t_sim = np.arange(n_steps, dtype=np.float64) * dt_sim

    u_bic = 0.25 * np.sin(2.0 * np.pi * t_sim / t_sim[-1]).clip(min=0.0)
    u_tri = 0.25 * (-np.sin(2.0 * np.pi * t_sim / t_sim[-1])).clip(min=0.0)

    q_hist = np.empty((n_steps, model.nq), dtype=np.float64)
    u_true = np.column_stack([u_bic, u_tri])
    data.qpos[0] = 0.0
    data.act[:] = u_true[0]
    for k in range(n_steps):
        q_hist[k] = data.qpos.copy()
        data.ctrl[:] = u_true[k]
        mujoco.mj_step(model, data)

    stride = 5
    kin_time = t_sim[::stride]
    q_clip = q_hist[::stride]
    u_clip = u_true[::stride]

    cmap = EMGChannelMap(
        name="arm_2ch",
        bindings=(
            ChannelBinding("biceps", ("biceps_r",)),
            ChannelBinding("triceps", ("triceps_r",)),
        ),
    )
    good_trial = EMGTrial(
        trial_id="nominal_flexion",
        time=kin_time,
        signals=u_clip,
        channel_names=("biceps", "triceps"),
        signal_kind="mvc_normalized",
        qpos=q_clip,
        kinematics_time=kin_time,
    )

    rep_good = validate_trial(
        model,
        good_trial,
        channel_map=cmap,
        trim_s=0.05,
        tolerance_band=0.20,
        include_rollout_audit=True,
        uncertainty=UncertaintyConfig(
            samples=6, joint_noise_deg=0.2, strength_cv=0.1, mass_cv=0.03, seed=7
        ),
    )
    assert rep_good["flags"] == []
    assert rep_good["inverse_emg"]["per_channel"]["biceps"]["r_opt_lag"] > 0.85
    assert "band_coverage_mean" in rep_good["inverse_emg"]
    assert rep_good["feasibility"]["emg_bounded_reserve_ratio_p95"] < 0.12
    assert rep_good["open_loop_drift"]["one_step_mae_deg"] < 1.0

    # Known-bad mutation 1: phase-inverted EMG waveform
    swapped_signals = np.column_stack([1.0 - u_clip[:, 0], 0.85 * np.ones(len(u_clip))])
    bad_emg_trial = EMGTrial(
        trial_id="phase_inverted_emg",
        time=kin_time,
        signals=swapped_signals,
        channel_names=("biceps", "triceps"),
        signal_kind="mvc_normalized",
        qpos=q_clip,
        kinematics_time=kin_time,
    )
    rep_bad_emg = validate_trial(
        model,
        bad_emg_trial,
        channel_map=cmap,
        trim_s=0.05,
        tolerance_band=0.15,
    )
    assert "poor_waveform_correlation" in rep_bad_emg["flags"]
    assert "emg_infeasible_reserves" in rep_bad_emg["flags"]
    assert (
        rep_bad_emg["feasibility"]["emg_bounded_reserve_ratio_p95"]
        > 5.0 * rep_good["feasibility"]["emg_bounded_reserve_ratio_p95"]
    )

    # Known-bad mutation 2: 20x weakened biceps fails EMG feasibility
    m_weak = mujoco.MjModel.from_xml_string(
        _TWO_MUSCLE_ARM_XML.replace(
            'name="biceps_r" tendon="bic_tendon" force="120"',
            'name="biceps_r" tendon="bic_tendon" force="5"',
        )
    )
    rep_weak = validate_trial(
        m_weak, good_trial, channel_map=cmap, trim_s=0.05, tolerance_band=0.10
    )
    assert "emg_infeasible_reserves" in rep_weak["flags"]

    # Known-bad mutation 3: severed / zero-force triceps triggers dead_actuators
    m_dead = mujoco.MjModel.from_xml_string(
        _TWO_MUSCLE_ARM_XML.replace(
            'name="triceps_r" tendon="tri_tendon" force="120"',
            'name="triceps_r" tendon="tri_tendon" force="0.0"',
        )
    )
    sens_dead = actuator_pose_sensitivity(m_dead, q_clip, max_states=8)
    assert sens_dead["n_near_zero"] == 1
    assert sens_dead["dead_actuators"] == ["triceps_r"]
    rep_dead = validate_trial(
        m_dead, good_trial, channel_map=cmap, include_rollout_audit=True
    )
    assert "dead_actuators" in rep_dead["flags"]

    agg = summarize_emg_validation([rep_good, rep_bad_emg])
    assert agg["trials"] == 2
    assert len(agg["flagged"]) == 1
    assert agg["flagged"][0]["trial_id"] == "phase_inverted_emg"


@needs_myo_sim_fingerprint
def test_rsa_and_paired_wilcoxon_comparison() -> None:
    three_muscle_xml = _TWO_MUSCLE_ARM_XML.replace(
        '<muscle name="triceps_r" tendon="tri_tendon" force="120" lengthrange="0.09 0.20"/>',
        '<muscle name="triceps_r" tendon="tri_tendon" force="120" lengthrange="0.09 0.20"/>\n'
        '    <muscle name="brachialis_r" tendon="bic_tendon" force="60" lengthrange="0.09 0.20"/>',
    )
    model_a = mujoco.MjModel.from_xml_string(three_muscle_xml)
    model_b = mujoco.MjModel.from_xml_string(
        three_muscle_xml.replace('force="120"', 'force="15"', 1)
    )
    cmap = EMGChannelMap(
        name="arm_3ch",
        bindings=(
            ChannelBinding("bic", ("biceps_r",)),
            ChannelBinding("bra", ("brachialis_r",)),
            ChannelBinding("tri", ("triceps_r",)),
        ),
    )

    reports_a, reports_b = [], []
    t = np.linspace(0.0, 0.8, 81)
    for idx in range(6):
        amp = 0.20 + 0.03 * idx
        q = (amp * np.sin(2.0 * np.pi * t / 0.8))[:, None]
        flex = np.clip(np.sin(2.0 * np.pi * t / 0.8), 0.0, 1.0) * 0.35
        ext = np.clip(-np.sin(2.0 * np.pi * t / 0.8), 0.0, 1.0) * 0.35
        trial = EMGTrial(
            trial_id=f"S{idx:02d}_flex",
            time=t,
            signals=np.column_stack([flex, 0.8 * flex + 0.02, ext]),
            channel_names=("bic", "bra", "tri"),
            signal_kind="mvc_normalized",
            qpos=q,
            meta={"subject": f"S{idx:02d}"},
        )
        reports_a.append(validate_trial(model_a, trial, channel_map=cmap))
        reports_b.append(validate_trial(model_b, trial, channel_map=cmap))

    assert "muscle_rsa_spearman" in reports_a[0]["inverse_emg"]
    agg_a = summarize_emg_validation(reports_a)
    assert "muscle_rsa_spearman" in agg_a["stats"]

    comp = compare_emg_validation_reports(
        reports_a, reports_b, label_a="nominal", label_b="weakened"
    )
    assert comp["unit_name"] == "subject"
    assert comp["shared_units"] == 6
    metrics_tested = {row["metric"] for row in comp["summary_rows"]}
    assert "r_opt_lag" in metrics_tested
    assert "emg_bounded_reserve_ratio_p95" in metrics_tested
    assert len(comp["per_channel_rows"]) == 3
    comp_bridge = compare_emg_reports_mm_evalkit(
        reports_a, reports_b, label_a="nominal", label_b="weakened"
    )
    assert comp_bridge["shared_units"] == 6


def test_export_mm_evalkit_rollout_contracts(tmp_path: Path) -> None:
    model_fixed = mujoco.MjModel.from_xml_string(_TWO_MUSCLE_ARM_XML)
    t_src = np.linspace(0.0, 0.4, 81)
    q_fixed = (0.25 * np.sin(2.0 * np.pi * t_src / 0.4))[:, None]
    trial_fixed = EMGTrial(
        trial_id="ep0",
        time=t_src,
        signals=np.column_stack([np.abs(q_fixed[:, 0]), 0.1 * np.ones_like(t_src)]),
        channel_names=("biceps", "triceps"),
        qpos=q_fixed,
    )

    so_npz = export_mm_evalkit_rollout(
        model_fixed,
        [trial_fixed],
        tmp_path / "fixed_so.npz",
        solver="static_optimization",
        motion_group="arm_group",
    )
    cmc_npz = export_mm_evalkit_rollout(
        model_fixed,
        [trial_fixed],
        tmp_path / "fixed_cmc.npz",
        solver="cmc",
        motion_group="arm_group",
    )
    policy_yaml = write_mm_evalkit_policy_config(
        tmp_path / "policies_fixed.yaml",
        [
            {"id": "Model SO", "label": "SO Model", "trajectory_data_path": so_npz},
            {"id": "Model CMC", "label": "CMC Model", "trajectory_data_path": cmc_npz},
        ],
    )

    with np.load(so_npz, allow_pickle=False) as d:
        assert int(d["n_episodes"]) == 1
        assert float(d["dt"]) == pytest.approx(0.01)
        assert str(d["activation_sample_phase"]) == "pre_step"
        assert list(d["joint_names"]) == ["elbow_flexion"]
        assert d["episode_0_joint_positions"].shape == (41, 1)
        assert d["episode_0_muscle_activations"].shape == (41, 2)
        assert d["episode_0_touch_observations"].shape == (41, 4)

    free_xml = """
    <mujoco model="free_walker">
      <option timestep="0.002" gravity="0 0 -9.81"/>
      <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1"/>
        <body name="pelvis" pos="0 0 0.9">
          <freejoint name="root"/>
          <geom type="sphere" size="0.08" mass="5.0"/>
          <body name="shank_r" pos="0 -0.1 -0.4">
            <joint name="knee_r" type="hinge" axis="0 1 0" range="-1.5 0.2"/>
            <geom name="foot_col1_r" type="sphere" pos="0 0 -0.45" size="0.05" mass="1.0"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <muscle name="quad_r" joint="knee_r" gear="20" force="300" lengthrange="-0.5 0.5"/>
      </actuator>
    </mujoco>
    """
    model_free = mujoco.MjModel.from_xml_string(free_xml)
    t_gait = np.linspace(0.0, 1.0, 101)
    q_free = np.tile(model_free.qpos0, (len(t_gait), 1))
    q_free[:, 7] = -0.2 + 0.1 * np.sin(2.0 * np.pi * t_gait)
    trial_free = EMGTrial(
        trial_id="walk_0",
        time=t_gait,
        signals=0.3 * np.ones((len(t_gait), 1)),
        channel_names=("quad",),
        qpos=q_free,
        cycle_events=np.array([[0.0, 1.0]]),
    )
    gait_a = export_mm_evalkit_rollout(
        model_free, [trial_free], tmp_path / "gait_a.npz", motion_group="walk_set"
    )
    gait_b = export_mm_evalkit_rollout(
        model_free, [trial_free], tmp_path / "gait_b.npz", motion_group="walk_set"
    )
    gait_yaml = write_mm_evalkit_policy_config(
        tmp_path / "policies_gait.yaml",
        [
            {"id": "walk-a", "label": "Walk A", "trajectory_data_path": gait_a},
            {"id": "walk-b", "label": "Walk B", "trajectory_data_path": gait_b},
        ],
    )

    try:
        from mm_evalkit.policies import load_policy_specs, validate_policy_files
        from mm_evalkit.workflows.bimanual.rollout import open_rollout
        from mm_evalkit.workflows.gait.rollout_validation import (
            validate_policy_rollouts,
        )
    except ImportError:
        return

    specs_fixed = load_policy_specs(policy_yaml)
    validate_policy_files(specs_fixed)
    for spec in specs_fixed:
        with open_rollout(
            spec.trajectory_path,
            expected_activation_count=2,
            expected_motion_group="arm_group",
        ) as (_, meta):
            assert meta.n_episodes == 1
            assert meta.joint_names == ("elbow_flexion",)

    specs_gait = load_policy_specs(gait_yaml)
    validate_policy_files(specs_gait)
    validate_policy_rollouts(
        specs_gait,
        subject_based=True,
        motion_count=1,
        expected_joint_names=("root", "knee_r"),
        expected_joint_width=model_free.nq,
        expected_activation_width=model_free.nu,
        expected_motion_group="walk_set",
    )


def test_emg_trial_supported_dofs_roundtrip_and_auto_forwarding(
    tmp_path: Path,
) -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><option gravity="0 0 0"/><worldbody><body>
          <joint name="elbow" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02" mass="1"/>
          <body pos=".3 0 0"><joint name="wrist" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 .08 0 0" size=".02" mass=".4"/>
          </body></body></worldbody>
          <tendon><fixed name="fcr"><joint joint="elbow" coef=".01"/>
            <joint joint="wrist" coef=".02"/></fixed></tendon>
          <actuator>
            <motor name="FCR" tendon="fcr" gear="500" ctrlrange="0 1"/>
            <motor name="TRI" joint="elbow" gear="-20" ctrlrange="0 1"/>
            <motor name="ECR" joint="wrist" gear="-10" ctrlrange="0 1"/>
          </actuator></mujoco>"""
    )
    t = np.linspace(0.0, 0.5, 51)
    trial = EMGTrial(
        trial_id="strapped_wrist_trial",
        time=t,
        signals=np.column_stack([np.full_like(t, 0.5), np.zeros_like(t)]),
        channel_names=("FCR", "TRI"),
        signal_kind="mvc_normalized",
        qpos=np.zeros((len(t), 2)),
        kinematics_time=t,
        external_qfrc=np.tile(np.array([[0.0, -5.0]]), (len(t), 1)),
        supported_dofs=(0,),
    )
    saved = tmp_path / "strapped.npz"
    save_emg_trial(saved, trial)
    loaded = load_emg_trial(saved)
    assert loaded.supported_dofs == (0,)

    cmap = EMGChannelMap(
        name="two_ch",
        bindings=(
            ChannelBinding(channel="FCR", actuators=("FCR",)),
            ChannelBinding(channel="TRI", actuators=("TRI",)),
        ),
    )
    rep = validate_trial(model, loaded, channel_map=cmap, trim_s=0.0)
    assert rep["inverse_emg"]["per_channel"]["TRI"]["peak_pred"] == pytest.approx(
        0.0, abs=1e-4
    )
    assert rep["effort"]["reserve_to_torque_ratio_p95"] == pytest.approx(0.0, abs=1e-4)
