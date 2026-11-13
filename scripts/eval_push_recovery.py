# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Paired CPU push grids with explicit physical dose, provenance and uncertainty."""

from __future__ import annotations

import argparse
import itertools
import inspect
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from myosuite import make_env
from myosuite.envs.wrappers import PerturbationWrapper
from myosuite.utils.checkpoint_utils import find_vec_normalize, load_policy
from myosuite.utils.deployment_manifest import (
    artifact_record,
    env_deployment_manifest,
    model_identity,
)
from myosuite.utils.paired_replay import ReplayPolicy
from myosuite.utils.push_metrics import RecoveryCriteria
from myosuite.utils.push_protocol import PushProtocol
from myosuite.utils.push_recovery import run_paired_push
from myosuite.utils.push_results import (
    save_trial,
    summarize_conditions,
    write_json,
    write_trials_csv,
)
from myosuite.utils.push_scheduling import ContactPhase


@dataclass(frozen=True)
class StudyConfig:
    """JSON-configured Cartesian conditions; only initial-state seeds replicate."""

    env_id: str
    protocol: PushProtocol
    checkpoint: str | None = None
    checkpoint_source: dict[str, str] | None = None
    seeds: tuple[int, ...] = (0,)
    directions: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: {"lateral": (1, 0, 0)}
    )
    forces_bw: tuple[float, ...] = (0.1,)
    phases: tuple[float | None, ...] = (None,)
    contact_phase: ContactPhase | None = None
    velocity_jitter_natural: float = 0.0
    criteria: RecoveryCriteria = RecoveryCriteria()
    statistics_seed: int = 0
    bootstrap_resamples: int = 2000

    def __post_init__(self) -> None:
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(
                not isinstance(s, int) or isinstance(s, bool) or s < 0
                for s in self.seeds
            )
        ):
            raise ValueError("Use distinct nonnegative integer seeds")
        if not self.directions or not self.forces_bw or not self.phases:
            raise ValueError("Condition dimensions must be nonempty")
        if not self.velocity_jitter_natural and len(self.seeds) != 1:
            raise ValueError(
                "A deterministic sweep uses one seed; declare jitter for replicated trials"
            )
        if self.contact_phase is not None and any(p is None for p in self.phases):
            raise ValueError("Contact-phase grids require numeric phases")

    @classmethod
    def read(cls, path: Path) -> StudyConfig:
        """Load typed configurations, rejecting unknown constructor fields."""
        data = json.loads(path.read_text(encoding="utf-8"))
        data["protocol"] = PushProtocol(**data["protocol"])
        if data.get("contact_phase") is not None:
            data["contact_phase"] = ContactPhase(**data["contact_phase"])
        if "criteria" in data:
            data["criteria"] = RecoveryCriteria(**data["criteria"])
        return cls(**data)


def run_study(
    config: StudyConfig, output: Path, *, checkpoint: Path | None = None
) -> dict[str, Any]:
    """Run one explicit design, closing its env after success or failure.

    Existing output directories are protected. A complete.json publication marker
    appears only after every trace, row and summary is written. Partial directories
    from interrupted runs are deliberately not accepted as reusable results.
    """
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(output)
    source = None
    if checkpoint is None and config.checkpoint is not None:
        checkpoint = Path(config.checkpoint)
    if checkpoint is None:
        from huggingface_hub import hf_hub_download

        source = config.checkpoint_source
        if source is None:
            raise ValueError("Provide a local checkpoint or pinned checkpoint_source")
        if source["revision"].lower() in ("main", "master", "head"):
            raise ValueError("Checkpoint downloads require an immutable tag or commit")
        checkpoint = Path(hf_hub_download(**source))
    if not checkpoint.is_file() or checkpoint.suffix not in (".zip", ".pt"):
        raise ValueError("Use one explicit SB3 .zip or RSL-RL .pt checkpoint file")
    checkpoint_record = artifact_record(checkpoint, source=source)
    download_s = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=False)
    env = make_env(config.env_id)
    environment_ready = time.perf_counter()
    try:
        policy = ReplayPolicy.stateless(load_policy(env, checkpoint, strict=True))
        context = {}
        write_json(output / "config.json", asdict(config))
        context["study_config"] = artifact_record(output / "config.json")
        for owner in (
            StudyConfig,
            ReplayPolicy,
            PushProtocol,
            ContactPhase,
            RecoveryCriteria,
            run_paired_push,
            save_trial,
            PerturbationWrapper,
        ):
            path = Path(inspect.getfile(owner))
            context["source_" + path.name] = artifact_record(path)
        stats = find_vec_normalize(checkpoint) if checkpoint.suffix == ".zip" else None
        if stats is not None:
            context["frozen_vecnormalize"] = artifact_record(stats)
        for name in ("agent.yaml", "env.yaml"):
            path = checkpoint.parent / "params" / name
            if path.is_file():
                context[name] = artifact_record(path)
        manifest = env_deployment_manifest(
            env,
            checkpoint=checkpoint_record,
            references=context,
            normalization={
                "kind": "existing_loader_frozen_statistics",
                "source": "checkpoint_or_recorded_vecnormalize",
                "details": "unknown_when_not_recorded_by_legacy_checkpoint",
            },
            runtime={
                "policy_mode": "deterministic_mean",
                "controller_state": "stateless_feedforward",
            },
        )
        write_json(output / "deployment.json", manifest)
        policy_ready = time.perf_counter()
        rows, rollout_s, artifact_s = [], 0.0, 0.0
        conditions = itertools.product(
            config.directions.items(), config.phases, config.forces_bw
        )
        for condition, ((direction_name, direction), phase, force) in enumerate(
            conditions
        ):
            protocol = replace(
                config.protocol, direction=direction, force_n=None, force_bw=force
            )
            contact = None
            if config.contact_phase is not None:
                contact = replace(config.contact_phase, phase=phase)
            elif phase is not None:
                protocol = replace(protocol, start_s=None, phase=phase)
            name = f"{condition:03d}:{direction_name}:phase={phase}:bw={force}"
            for seed in config.seeds:
                before = time.perf_counter()
                trial = run_paired_push(
                    env,
                    policy,
                    protocol,
                    seed=seed,
                    contact_phase=contact,
                    velocity_jitter_natural=config.velocity_jitter_natural,
                )
                elapsed = time.perf_counter() - before
                rollout_s += elapsed
                before = time.perf_counter()
                rows.append(
                    save_trial(
                        output,
                        len(rows),
                        name,
                        trial,
                        config.criteria,
                        elapsed_s=elapsed,
                    )
                )
                rows[-1]["model"] = model_identity(env.unwrapped.model)
                artifact_s += time.perf_counter() - before
        write_json(output / "trials.json", rows)
        write_trials_csv(output / "trials.csv", rows)
        summary = {
            "schema": 1,
            "env_id": config.env_id,
            "experiment": "seeded_initial_velocity_jitter"
            if config.velocity_jitter_natural
            else "deterministic_sweep",
            "quality_scope": "paired_baseline_recovery_and_explicit_command_diagnostics; not_training_convergence",
            "trials": len(rows),
            "statistics": summarize_conditions(
                rows,
                independent_jitter=config.velocity_jitter_natural > 0,
                seed=config.statistics_seed,
                resamples=config.bootstrap_resamples,
            ),
            "timing": {
                "scope": "run_study before final summary/complete marker writes; interpreter/import startup excluded",
                "cache_state": "not_measured",
                "checkpoint_resolve_s": download_s,
                "environment_startup_s": environment_ready - started - download_s,
                "policy_load_and_provenance_s": policy_ready - environment_ready,
                "paired_rollout_s": rollout_s,
                "trial_artifact_io_s": artifact_s,
                "wall_s": time.perf_counter() - started,
            },
        }
        write_json(output / "summary.json", summary)
        write_json(
            output / "complete.json",
            {
                "schema": 1,
                "artifacts": {
                    name: artifact_record(output / name)
                    for name in (
                        "config.json",
                        "deployment.json",
                        "trials.json",
                        "trials.csv",
                        "summary.json",
                    )
                },
            },
        )
        return summary
    finally:
        env.close()


def main() -> None:
    """Evaluate a JSON-configured study with the existing checkpoint adapters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--velocity-jitter-natural", type=float)
    args = parser.parse_args()
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    import torch

    torch.set_num_threads(args.torch_threads)
    overrides = {}
    if args.seeds is not None:
        overrides["seeds"] = tuple(args.seeds)
    if args.velocity_jitter_natural is not None:
        overrides["velocity_jitter_natural"] = args.velocity_jitter_natural
    config = replace(StudyConfig.read(args.config), **overrides)
    result = run_study(config, args.output, checkpoint=args.checkpoint)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "trials": result["trials"],
                "experiment": result["experiment"],
            }
        )
    )


if __name__ == "__main__":
    main()
