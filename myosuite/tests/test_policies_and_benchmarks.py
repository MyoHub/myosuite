# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for ``myosuite.policies``, ``myosuite.benchmarks``, and legacy CLI shims."""

from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.tier1


def test_policies_and_benchmarks_lazy_facades_and_preserved_exports() -> None:
    import myosuite.benchmarks as benchmarks
    import myosuite.core as core
    import myosuite.policies as policies

    assert (
        policies.ActionMap
        is importlib.import_module("myosuite.policies.action_map").ActionMap
    )
    assert (
        policies.load_policy
        is importlib.import_module("myosuite.policies.checkpoint_utils").load_policy
    )
    assert policies.export_onnx is importlib.import_module(
        "myosuite.policies.export_onnx"
    )
    assert benchmarks.examine_env is importlib.import_module(
        "myosuite.benchmarks.examine_env"
    )
    assert benchmarks.term_catalog is importlib.import_module(
        "myosuite.benchmarks.term_catalog"
    )

    # Preserved core viewer loop exports (Finding #3)
    assert callable(core.run_passive_viewer_loop)
    assert core.StepCallback is not None
    assert core.MetricCallback is not None

    # Preserved MuscleMimic runner import path when mjlab/torch are installed
    try:
        runner_mod = importlib.import_module(
            "myosuite.integrations.musclemimic.mjlab_policy_runner"
        )
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] not in {"torch", "mjlab"}:
            raise
    else:
        assert hasattr(runner_mod, "OnnxCheckpointingMjlabRunner")


@pytest.mark.parametrize(
    ("module_name", "flag", "expected_snippet"),
    [
        ("myosuite.terms.catalog", "--help", "--kind"),
        ("myosuite.utils.export_onnx", "--help", "export"),
        ("myosuite.utils.examine_env", "--help", "--env_name"),
        ("myosuite.utils.examine_sim", "--help", "--sim_path"),
        ("myosuite.utils.path_plotting", "--help", "--path"),
        ("myosuite.logger.examine_logs", "--help", "--env_name"),
        ("myosuite.logger.examine_reference", "--help", "--env_name"),
        ("myosuite.integrations.musclemimic.fullbody_eval_cli", "--help", "--path"),
    ],
)
def test_legacy_cli_module_shims_forward_help(
    module_name: str, flag: str, expected_snippet: str
) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", module_name, flag],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (
        proc.returncode == 0
    ), f"{module_name}: rc={proc.returncode}, stderr={proc.stderr}"
    assert expected_snippet in proc.stdout


def test_legacy_dict_utils_main_runs_unit_check() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "myosuite.utils.dict_utils"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stderr


def test_write_manifest_records_musclemimic_version_in_fresh_process(
    tmp_path: Path,
) -> None:
    """``write_manifest(..., contract={})`` records ``musclemimic_models_version`` without importing ``model_versions`` explicitly."""
    code = f"""
import json
from pathlib import Path
from myosuite.policies.checkpoint_manifest import write_manifest
d = Path({str(tmp_path)!r})
try:
    write_manifest(d, "myoMimicFullbody-v0", contract={{}})
except RuntimeError as exc:
    assert "not installed" in str(exc)
else:
    data = json.loads((d / "manifest.json").read_text())
    assert "musclemimic_models_version" in data
"""
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr


def test_single_manifest_provider_and_mjlab_experiment_provider() -> None:
    """``model_versions`` does not duplicate ``_MANIFEST_PROVIDERS``, and ``envs`` registers ``mjlab_experiment``."""
    import myosuite  # noqa: F401
    import myosuite.integrations.musclemimic.model_versions  # noqa: F401
    from myosuite.policies import checkpoint_manifest, checkpoint_utils

    assert len(checkpoint_manifest._MANIFEST_PROVIDERS) == 1
    assert len(checkpoint_utils._MJLAB_EXPERIMENT_PROVIDERS) == 1


def test_import_myosuite_exposes_lazy_utils_attribute_in_fresh_process() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import myosuite; assert callable(myosuite.utils.seed_envs) and hasattr(myosuite.utils, 'gym') and callable(myosuite.utils.path_utils.evaluate_success) and myosuite.utils.policy_utils is not None and myosuite.logger.grouped_datasets is not None",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
