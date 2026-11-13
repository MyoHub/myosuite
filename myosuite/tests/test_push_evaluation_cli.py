# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""CLI orchestration through a real SB3 adapter and registered native CPU task."""

import json
from dataclasses import replace

import numpy as np
import pytest

from myosuite import make_env
from myosuite.utils.push_protocol import PushProtocol
from myosuite.utils.push_metrics import RecoveryCriteria
from scripts.eval_push_recovery import StudyConfig, run_study

pytestmark = pytest.mark.tier1


def test_real_sb3_native_zero_control_publishes_reproducible_artifacts(
    tmp_path,
) -> None:
    sb3 = pytest.importorskip("stable_baselines3")
    env_id = "myoElbowPose1D6MFixed-v0"
    env = make_env(env_id)
    try:
        body = env.unwrapped.model.body(int(env.unwrapped.model.jnt_bodyid[0])).name
        checkpoint = tmp_path / "policy.zip"
        sb3.PPO("MlpPolicy", env, n_steps=8, batch_size=8, seed=0, device="cpu").save(
            checkpoint
        )
    finally:
        env.close()
    cfg = StudyConfig(
        env_id,
        PushProtocol(body, 0.4, steps=6, start_s=0, duration_s=0.01),
        forces_bw=(0,),
        criteria=RecoveryCriteria(min_height_L=0, hold_s=0.01),
    )
    output = tmp_path / "trial"
    result = run_study(cfg, output, checkpoint=checkpoint)
    assert result["trials"] == 1 and result["experiment"] == "deterministic_sweep"
    assert (output / "complete.json").is_file()
    rows = json.loads((output / "trials.json").read_text())
    assert rows[0]["metrics"]["max_speed_error_natural"] == 0
    with np.load(output / "trial_00000.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(
            arrays["baseline_actions"], arrays["pushed_actions"]
        )
    deployment = json.loads((output / "deployment.json").read_text())
    assert deployment["checkpoint"]["sha256"]
    assert deployment["model"]["physical_sha256"]
    with pytest.raises(FileExistsError):
        run_study(cfg, output, checkpoint=checkpoint)


def test_config_rejects_fake_independent_seeds_and_accepts_explicit_jitter() -> None:
    cfg = StudyConfig("test", PushProtocol("pelvis", 0.9))
    with pytest.raises(ValueError, match="deterministic"):
        replace(cfg, seeds=(0, 1, 2))
    assert replace(cfg, seeds=(0, 1, 2), velocity_jitter_natural=0.02).seeds == (
        0,
        1,
        2,
    )
