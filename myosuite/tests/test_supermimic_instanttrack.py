# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""InstantTrack additions to myoSuperMimic (CPU, no simulator)."""

import pytest
import torch
from torch import nn

from myosuite.learning.supermimic.instanttrack import (
    FailureResets,
    FootContact,
    MuonAdam,
    foot_site_indices,
    make_optimizer,
    policy_lr_multiplier,
)

pytestmark = pytest.mark.tier2


def test_failure_resets_bins_and_previous_bin():
    fr = FailureResets([250, 120], bin_frames=100, share=1.0, device="cpu")
    # clip 0: bins 0,1,2 (250 frames); clip 1: bins 3,4
    assert fr.bin_clip.tolist() == [0, 0, 0, 1, 1]
    assert fr.bin_start.tolist() == [0, 100, 200, 0, 100]
    assert fr.bin_prev.tolist() == [0, 0, 1, 3, 3]
    assert fr.bins(torch.tensor([0, 0, 1]), torch.tensor([99, 1000, 105])).tolist() == [
        0,
        2,
        4,
    ]


def test_failure_resets_sample_before_failures_of_own_clip():
    torch.manual_seed(0)
    fr = FailureResets([300, 300], bin_frames=100, share=1.0, device="cpu")
    offsets = torch.full((400,), 7)
    # nothing failed yet: offsets unchanged
    assert torch.equal(fr.sample(offsets, torch.zeros(400, dtype=torch.long)), offsets)
    # clip 0 fails in its bin 2 (frames 200-299); truncations in bin 1 do not count
    fr.update(
        torch.tensor([0, 0, 0]),
        torch.tensor([250, 260, 150]),
        torch.tensor([True, True, False]),
    )
    out = fr.sample(offsets, torch.zeros(400, dtype=torch.long))
    assert ((out >= 100) & (out < 200)).all()  # one bin before the failure
    # an env of clip 1 has no failure of its own clip: keeps its default start
    out1 = fr.sample(offsets, torch.ones(400, dtype=torch.long))
    assert torch.equal(out1, offsets)
    # single-clip envs pass clip=None
    fr1 = FailureResets([300], bin_frames=100, share=0.5, device="cpu")
    fr1.update(
        torch.tensor([0]), torch.tensor([10]), torch.tensor([True])
    )  # bin 0: previous bin is itself
    out = fr1.sample(torch.full((2000,), 250), None)
    moved = out != 250
    assert 0.4 < moved.float().mean() < 0.6
    assert (out[moved] < 100).all()


def test_failure_rates_fade():
    fr = FailureResets([100], bin_frames=50, share=0.3, device="cpu")
    fr.update(torch.tensor([0]), torch.tensor([60]), torch.tensor([True]))
    v = float(fr.fail[1])
    for _ in range(10):
        fr.update(
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.bool),
        )
    assert float(fr.fail[1]) < v


def test_foot_contact_reward():
    T, dt = 50, 0.02
    sites = torch.zeros(T, 3, 3)
    sites[:, 0, 2] = 0.05  # foot site 0 on the ground at z = 0.05 (its floor)
    sites[25:, 0, 2] = 0.30  # then lifted
    sites[:, 1, 2] = 1.0  # not a foot
    sites[:, 2, 2] = 0.05
    sites[:, 2, 0] = (
        torch.arange(T) * 0.1
    )  # foot site 2 slides at 5 m/s: not a reference contact
    fc = FootContact(sites, [0, 2], dt)
    assert fc.ref[:25, 0].eq(1).all() and fc.ref[25:, 0].eq(0).all()
    assert fc.ref[:, 1].eq(0).all()
    f = torch.tensor([5, 40])
    # matching the reference without slip: 2 for a down foot, 1 for a lifted one
    r = fc.reward(sites[f], sites[f - 1], f)
    assert r.shape == (2,)
    good = fc.reward(sites[f][:1], sites[f][:1], f[:1])
    lifted = sites[f][:1].clone()
    lifted[:, 0, 2] = 0.3
    bad = fc.reward(lifted, lifted, f[:1])
    assert float(good) > float(bad)


def test_foot_site_names():
    names = [
        "pelvis_mimic",
        "left_ankle_mimic",
        "left_toes_mimic",
        "head_mimic",
        "right_toes_mimic",
    ]
    assert foot_site_indices(names) == [1, 2, 4]


def test_muon_adam_splits_and_steps():
    net = nn.Sequential(
        nn.Linear(4, 8), nn.ELU(), nn.Linear(8, 8), nn.ELU(), nn.Linear(8, 2)
    )
    opt = make_optimizer("muon", net, 1e-2)
    assert isinstance(opt, MuonAdam) and len(opt.opts) == 2
    muon_params = opt.opts[1].param_groups[0]["params"]
    assert len(muon_params) == 1 and muon_params[0] is net[2].weight
    before = [p.detach().clone() for p in net.parameters()]
    loss = net(torch.randn(16, 4)).square().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert all(not torch.equal(a, b) for a, b in zip(before, net.parameters()))
    for g in opt.param_groups:
        g["lr"] = 5e-3
    sd = opt.state_dict()
    opt2 = make_optimizer("muon", net, 1e-2)
    opt2.load_state_dict(sd)
    assert all(g["lr"] == 5e-3 for g in opt2.param_groups)


def test_policy_lr_multiplier():
    assert policy_lr_multiplier(0, 4.0, 100, 50) == 4.0
    assert policy_lr_multiplier(100, 4.0, 100, 50) == 4.0
    assert policy_lr_multiplier(125, 4.0, 100, 50) == 2.5
    assert policy_lr_multiplier(10_000, 4.0, 100, 50) == 1.0
    assert policy_lr_multiplier(5, 1.0, 100, 50) == 1.0


def test_fullbody_model_source(monkeypatch):
    import importlib.util

    from myosuite.integrations.musclemimic.fullbody_model import (
        MIMIC_FULLBODY_MODEL_ENV,
        default_mimic_fullbody_config,
        mimic_fullbody_model_source,
    )

    monkeypatch.setenv(MIMIC_FULLBODY_MODEL_ENV, "myo_sim")
    assert mimic_fullbody_model_source() == "myo_sim"
    cfg = default_mimic_fullbody_config()
    cfg.model_source = "musclemimic"  # the config wins over the variable
    assert mimic_fullbody_model_source(cfg) == "musclemimic"
    monkeypatch.setenv(MIMIC_FULLBODY_MODEL_ENV, "auto")
    have = importlib.util.find_spec("musclemimic_models") is not None
    assert mimic_fullbody_model_source() == ("musclemimic" if have else "myo_sim")
    monkeypatch.setenv(MIMIC_FULLBODY_MODEL_ENV, "bogus")
    with pytest.raises(ValueError):
        mimic_fullbody_model_source()


def test_myo_sim_fullbody_takes_musclemimic_clip_joint_names():
    import numpy as np

    from myosuite.core.trajectory_io import MotionClip, expand_motion_clip_to_model
    from myosuite.integrations.musclemimic.fullbody_model import (
        build_native_mimic_fullbody_spec,
        default_mimic_fullbody_config,
    )

    m = build_native_mimic_fullbody_spec(default_mimic_fullbody_config()).compile()
    names = [m.joint(i).name for i in range(m.njnt)]
    assert "elbow_flexion_l" in names and "elbow_flex_l" not in names
    # a MuscleMimic-named clip: elbow_flex_* instead of myo_sim's elbow_flexion_*
    clip_names = [
        {"elbow_flexion_l": "elbow_flex_l", "elbow_flexion_r": "elbow_flex_r"}.get(n, n)
        for n in names
    ]
    qn = [n for j, n in enumerate(clip_names) for _ in range(7 if j == 0 else 1)]
    q = np.tile(m.qpos0, (3, 1))
    q[:, m.jnt_qposadr[names.index("elbow_flexion_l")]] = 0.7
    out = expand_motion_clip_to_model(
        MotionClip(
            qpos=q,
            qvel=None,
            site_xpos=None,
            site_names=None,
            qpos_joint_names=qn,
            frequency_hz=100.0,
        ),
        m,
    )
    assert np.allclose(out.qpos[:, m.jnt_qposadr[names.index("elbow_flexion_l")]], 0.7)
