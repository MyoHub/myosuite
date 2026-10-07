# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from myosuite.integrations.musclemimic.fullbody_native_playback import (
    parse_native_playback_argv,
)


def test_parse_native_playback_accepts_record_flags() -> None:
    args = parse_native_playback_argv(
        [
            "--path",
            "hf://repo/model",
            "--motion_path",
            "KIT/4/foo",
            "--use_mujoco",
            "--record",
            "--record_path",
            "out.mp4",
            "--record_fps",
            "30",
        ]
    )
    assert args.record is True
    assert args.record_path == "out.mp4"
    assert args.record_fps == 30
