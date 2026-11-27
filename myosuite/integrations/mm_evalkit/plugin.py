# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""``mm_evalkit.datasets`` entry-point runners delegating to the unified EMG workflow."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from myosuite.integrations.emg.trial_io import NpzDirectoryEMGSource
from myosuite.integrations.mm_evalkit.workflow import run_emg_validation


def _build_npz_directory_source(npz_dir: str) -> NpzDirectoryEMGSource:
    """Compatibility alias returning :class:`NpzDirectoryEMGSource`."""
    return NpzDirectoryEMGSource(npz_dir)


def _dispatch_dataset(dataset_key: str, argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    forwarded = (
        args
        if ("--dataset" in args or "--init-manifest" in args)
        else ["--dataset", dataset_key, *args]
    )
    return run_emg_validation(forwarded, prog=f"mm-evalkit evaluate {dataset_key}")


def _wheel_npz_main(argv: Sequence[str]) -> int:
    """Compatibility entry point evaluating ``--npz-dir`` through the unified workflow."""
    return _dispatch_dataset("custom", argv)


def rajagopal2016_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the Rajagopal et al. (2016) walking dataset."""
    return _dispatch_dataset("rajagopal2016", argv)


def lencioni2019_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the Lencioni et al. (2019) gait dataset."""
    return _dispatch_dataset("lencioni2019", argv)


def dimitrov2023_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the Dimitrov et al. (2013/2023) walking dataset."""
    return _dispatch_dataset("dimitrov2023", argv)


def camargo2021_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the Camargo et al. (2021) multi-terrain locomotion dataset."""
    return _dispatch_dataset("camargo2021", argv)


def custom_emg_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for custom manifest or standardized ``EMGTrial`` ``.npz`` directories."""
    return _dispatch_dataset("custom", argv)
