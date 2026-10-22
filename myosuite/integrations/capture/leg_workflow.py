# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Capture, retarget, train, resume and replay one registered MyoLeg subject."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import mujoco
import numpy as np

from myosuite import make_env
from myosuite.core.config import EnvConfig
from myosuite.core.subject_env import SubjectEnvConfig, subject_metadata
from myosuite.core.subject_model import (
    GroundAlignment,
    ScaleEstimate,
    SubjectModel,
    SubjectModelConfig,
    align_root_to_ground,
)
from myosuite.integrations.capture.markers import (
    MarkerTrial,
    SegmentMeasurement,
    estimate_marker_scales,
)
from myosuite.integrations.capture.retargeting import (
    load_retargeted_clip,
    retarget_markers,
)

ENV_ID = "myoLegWalk-v0"
MODULE = "myosuite.integrations.capture.leg_workflow"
SOLES = tuple(
    f"{name}_{side}"
    for side in "rl"
    for name in (
        "foot_col1",
        "foot_col3",
        "foot_col4",
        "bofoot_col1",
        "bofoot_col2",
    )
)
SITES = ("pelvis", "ASI_r", "ASI_l", "PSI_r", "PSI_l") + tuple(
    f"{name}_{side}"
    for side in "rl"
    for name in (
        "HJC",
        "TH1",
        "TH2",
        "LFC",
        "MFC",
        "KJC",
        "TB1",
        "TB2",
        "LMAL",
        "MMAL",
        "CAL",
        "TOE",
        "MT5",
        "toe",
        "edl-P6",
        "fhl-P7",
    )
)
KNOWN_FACTORS = {"femur_r": 1.05, "tibia_r": 1.03, "femur_l": 1.04, "tibia_l": 1.02}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _config(estimate: ScaleEstimate) -> SubjectEnvConfig:
    return SubjectEnvConfig(
        SubjectModelConfig(
            estimate,
            mass="keep",
            ground=GroundAlignment("root", SOLES, clearance=0.002),
        ),
        min_height=0.70,
        contact_margin=0.001,
    )


def _subject(env: Any) -> SubjectModel:
    return SubjectModel(
        env.unwrapped.model, env.unwrapped._mj_spec, subject_metadata(env)
    )


def _capture(model: mujoco.MjModel, qpos: np.ndarray, source: str) -> MarkerTrial:
    data = mujoco.MjData(model)
    ids = [model.site(name).id for name in SITES]
    frames = []
    for frame in qpos:
        data.qpos[:] = frame
        mujoco.mj_forward(model, data)
        frames.append(data.site_xpos[ids].copy())
    return MarkerTrial(
        SITES,
        np.array(frames),
        "m",
        np.eye(3),
        source,
        frequency_hz=100.0,
        definition_source="MyoLeg named anatomical sites; synthetic simulation",
        definition_license="Apache-2.0",
    )


def _synthetic_poses(model: mujoco.MjModel, ground: GroundAlignment) -> np.ndarray:
    frames = []
    for phase in (0.0, 0.5, 1.0):
        qpos = model.qpos0.copy()
        for side in "rl":
            qpos[model.joint(f"hip_flexion_{side}").qposadr[0]] += 0.01 * phase
            qpos[model.joint(f"knee_angle_{side}").qposadr[0]] += 0.04 * phase
        for _ in range(model.neq):
            for index in range(model.neq):
                if (
                    int(model.eq_type[index]) != int(mujoco.mjtEq.mjEQ_JOINT)
                    or not model.eq_active0[index]
                ):
                    continue
                first, second = model.eq_obj1id[index], model.eq_obj2id[index]
                if second >= 0:
                    x = (
                        qpos[model.jnt_qposadr[second]]
                        - model.qpos0[model.jnt_qposadr[second]]
                    )
                    qpos[model.jnt_qposadr[first]] = model.qpos0[
                        model.jnt_qposadr[first]
                    ] + np.polynomial.polynomial.polyval(x, model.eq_data[index, :5])
        for joint in range(model.njnt):
            if model.jnt_limited[joint] and int(model.jnt_type[joint]) in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                value = qpos[model.jnt_qposadr[joint]]
                if (
                    not model.jnt_range[joint, 0] - 1e-8
                    <= value
                    <= model.jnt_range[joint, 1] + 1e-8
                ):
                    raise ValueError(
                        f"Synthetic pose violates helper-joint limits: {model.joint(joint).name}"
                    )
        qpos, _ = align_root_to_ground(model, ground, qpos)
        frames.append(qpos)
    return np.array(frames)


def prepare(output: Path) -> dict[str, Any]:
    """Generate licensed synthetic capture, recover scales and finalize reset motion."""
    output.mkdir(parents=True, exist_ok=True)
    nominal = make_env(ENV_ID)
    truth = make_env(
        EnvConfig(
            ENV_ID,
            subject=_config(
                ScaleEstimate(
                    KNOWN_FACTORS, mass_kg=70.0, source="synthetic marker generator"
                )
            ),
        )
    )
    try:
        source_model, true_model = nominal.unwrapped.model, truth.unwrapped.model
        static = _capture(
            true_model,
            np.repeat(true_model.qpos0[None], 2, axis=0),
            "synthetic MyoLeg static trial",
        )
        reference = mujoco.MjData(source_model)
        mujoco.mj_forward(source_model, reference)
        measurements = []
        for side in "rl":
            for body, pair in (
                (f"femur_{side}", (f"HJC_{side}", f"LFC_{side}")),
                (f"tibia_{side}", (f"KJC_{side}", f"LMAL_{side}")),
            ):
                ids = [source_model.site(name).id for name in pair]
                if source_model.site_bodyid[ids[0]] != source_model.site_bodyid[ids[1]]:
                    raise ValueError(
                        "Scale-estimation markers must share their segment body."
                    )
                delta = reference.site_xpos[ids[1]] - reference.site_xpos[ids[0]]
                measurements.append(
                    SegmentMeasurement(
                        body,
                        pair,
                        float(np.linalg.norm(delta)),
                        expected_direction_model=delta,
                    )
                )
        estimated = estimate_marker_scales(static, measurements, mass_kg=70.0)
        for name, known in KNOWN_FACTORS.items():
            np.testing.assert_allclose(
                estimated.estimate.factors[name], [known] * 3, atol=1e-10
            )
        poses = _synthetic_poses(true_model, _config(estimated.estimate).model.ground)
        motion = _capture(true_model, poses, "synthetic MyoLeg moving trial")
        for name, trial in (("static", static), ("motion", motion)):
            _write(
                output / f"{name}_markers.json",
                {
                    "metadata": trial.metadata(),
                    "positions": np.asarray(trial.positions).tolist(),
                },
            )
        _write(
            output / "scale_estimate.json",
            {
                "estimate": estimated.estimate.to_dict(),
                "evidence": estimated.evidence,
                "synthetic_truth": KNOWN_FACTORS,
                "license": "Apache-2.0",
            },
        )
    finally:
        nominal.close()
        truth.close()
    config = _config(estimated.estimate)
    target = make_env(EnvConfig(ENV_ID, subject=config))
    try:
        fitted = retarget_markers(
            _subject(target),
            motion,
            dict(zip(SITES, SITES)),
            free_joint="root",
            sole_geoms=SOLES,
            tolerance_m=1e-4,
            scale_evidence=estimated.evidence,
        )
        config = replace(
            config,
            reset_qpos=tuple(fitted.clip.qpos[0]),
            reset_qvel=tuple(fitted.clip.qvel[0]),
        )
    finally:
        target.close()
    _write(output / "subject.json", config.to_dict())
    target = make_env(EnvConfig(ENV_ID, subject=config))
    try:
        subject = _subject(target)
        # The final model identity includes the authored reset keyframe.
        fitted = retarget_markers(
            subject,
            motion,
            dict(zip(SITES, SITES)),
            free_joint="root",
            sole_geoms=SOLES,
            tolerance_m=1e-4,
            scale_evidence=estimated.evidence,
        )
        clip_path = fitted.save(output / "retargeted_motion.npz")
        clip = load_retargeted_clip(clip_path, subject)
        target.reset(seed=7)
        np.testing.assert_allclose(target.unwrapped.data.qpos, clip.qpos[0], atol=1e-6)
        np.testing.assert_allclose(target.unwrapped.data.qvel, clip.qvel[0], atol=1e-6)
        report = {
            "env_id": ENV_ID,
            "license": "Apache-2.0; synthetic markers, no captured human data",
            "subject": str((output / "subject.json").resolve()),
            "subject_model": subject.metadata,
            "scale_estimate": estimated.estimate.to_dict(),
            "retargeted_motion": str(clip_path.resolve()),
            "quality": {
                name: value
                for name, value in fitted.diagnostics.items()
                if np.isscalar(value)
            },
        }
        _write(output / "preparation_report.json", report)
        return report
    finally:
        target.close()


def train(output: Path, *, resume: bool = False) -> Path:
    """Run the standard training entrypoint with a deliberately tiny CUDA PPO batch."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import torch
    import myosuite.envs.myo.backends.mjlab  # noqa: F401
    from scripts.train_mjlab import TrainConfig, run_train

    if not torch.cuda.is_available():
        raise RuntimeError(
            "The training stage requires a CUDA-enabled PyTorch environment."
        )
    config = SubjectEnvConfig.load(output / "subject.json")
    cfg = replace(TrainConfig.from_task(ENV_ID, subject=config), stop_on_success=False)
    cfg.env.scene.num_envs = 2
    cfg.env.episode_length_s = 0.2
    cfg.agent.num_steps_per_env = 4
    cfg.agent.max_iterations = 1
    cfg.agent.save_interval = 1
    cfg.agent.logger = "tensorboard"
    cfg.agent.upload_model = False
    cfg.agent.actor.hidden_dims = (32, 32)
    cfg.agent.critic.hidden_dims = (32, 32)
    cfg.agent.algorithm.num_learning_epochs = 1
    cfg.agent.algorithm.num_mini_batches = 1
    cfg.agent.resume = resume
    cfg.agent.load_run = "initial"
    directory = output / "runs" / ("resumed" if resume else "initial")
    if directory.exists():
        raise FileExistsError(
            f"Use a fresh output directory; training would overwrite {directory}."
        )
    run_train(ENV_ID, cfg, directory)
    checkpoint = max(
        directory.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[-1])
    )
    proof = {
        "device": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "num_envs": 2,
        "rollout_steps_per_env": 4,
        "updates_in_this_stage": 1,
        "checkpoint": str(checkpoint.resolve()),
    }
    if resume:
        from myosuite.utils.checkpoint_utils import resume_checkpoint

        previous = resume_checkpoint(directory.parent, ENV_ID, "initial")
        before = torch.load(previous, map_location="cpu", weights_only=False)[
            "actor_state_dict"
        ]
        after = torch.load(checkpoint, map_location="cpu", weights_only=False)[
            "actor_state_dict"
        ]
        delta = max(
            float((after[name] - tensor).abs().max())
            for name, tensor in before.items()
            if name.startswith("mlp.")
        )
        if delta <= 0:
            raise RuntimeError("Resumed PPO did not change actor parameters.")
        proof.update(
            resumed_from=str(previous.resolve()), actor_parameter_change_max=delta
        )
    _write(directory / "smoke_report.json", proof)
    return checkpoint


def replay(output: Path, *, steps: int = 20) -> dict[str, Any]:
    """Export the resumed policy, replay native/ONNX on CPU, and reject mismatches."""
    from myosuite.utils.checkpoint_manifest import check_contract
    from myosuite.utils.checkpoint_utils import load_policy
    from myosuite.utils.export_onnx import export_rslrl_to_onnx

    config = SubjectEnvConfig.load(output / "subject.json")
    checkpoints = output / "runs/resumed"
    checkpoint = max(
        checkpoints.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[-1])
    )
    target = make_env(EnvConfig(ENV_ID, subject=config))
    nominal = make_env(ENV_ID)
    changed = make_env(
        EnvConfig(ENV_ID, subject=replace(config, min_height=config.min_height + 0.01))
    )
    try:
        check_contract(ENV_ID, checkpoint.parent, env=target, strict=True)
        onnx = output / "policy.onnx"
        export_rslrl_to_onnx(
            checkpoint,
            onnx,
            target.observation_space.shape[0],
            target.action_space.shape[0],
        )
        rejected = []
        for description, env in (
            ("nominal morphology", nominal),
            ("different subject task", changed),
        ):
            for path in (checkpoint, onnx):
                try:
                    load_policy(env, path, strict=True)
                except ValueError as error:
                    if "morphology" not in str(error):
                        raise
                    rejected.append(f"{description}: {path.suffix}")
                else:
                    raise AssertionError(
                        f"Expected morphology admission rejection: {description} {path}"
                    )
        policies = {
            "native": load_policy(target, checkpoint, strict=True),
            "onnx": load_policy(target, onnx, strict=True),
        }
        initial_obs, _ = target.reset(seed=7)
        action_error = float(
            np.max(
                np.abs(policies["native"](initial_obs) - policies["onnx"](initial_obs))
            )
        )
        if action_error > 1e-5:
            raise RuntimeError(
                f"ONNX inference differs from the native actor by {action_error}."
            )
        values = {}
        for name, path in (("native", checkpoint), ("onnx", onnx)):
            policy = policies[name]
            obs, _ = target.reset(seed=7)
            reward_sum, episodes = 0.0, 0
            for _ in range(steps):
                obs, reward, terminated, truncated, _ = target.step(policy(obs))
                if not np.isfinite(obs).all() or not np.isfinite(reward):
                    raise RuntimeError("Policy replay produced a nonfinite state.")
                reward_sum += float(reward)
                if terminated or truncated:
                    episodes += 1
                    obs, _ = target.reset(seed=7 + episodes)
            values[name] = {
                "steps": steps,
                "reward_sum": reward_sum,
                "completed_episodes": episodes,
            }
        report = {
            "checkpoint": str(checkpoint.resolve()),
            "onnx": str(onnx.resolve()),
            "cpu_replay": values,
            "native_onnx_action_error_max": action_error,
            "training": {
                name: json.loads(
                    (output / "runs" / name / "smoke_report.json").read_text(
                        encoding="utf-8"
                    )
                )
                for name in ("initial", "resumed")
            },
            "mismatch_rejections": rejected,
            "subject_model": subject_metadata(target),
            "scope": "Pipeline smoke run; one PPO iteration and one resumed iteration do not establish walking performance.",
        }
        _write(output / "training_report.json", report)
        preparation = json.loads(
            (output / "preparation_report.json").read_text(encoding="utf-8")
        )
        training = json.loads(json.dumps(report["training"]))
        for stage in training.values():
            for key in ("checkpoint", "resumed_from"):
                if key in stage:
                    stage[key] = Path(stage[key]).relative_to(output).as_posix()
        metadata = subject_metadata(target)
        summary = {
            "schema": 1,
            "scope": report["scope"],
            "license": preparation["license"],
            "env_id": ENV_ID,
            "reproduce": f"python -m {MODULE} --stage all --output tasks/subject-leg-smoke --gpu-python <cuda-python>",
            "python": platform.python_version(),
            "dependencies": {
                name: version(name)
                for name in (
                    "torch",
                    "mujoco",
                    "mujoco-warp",
                    "mjlab",
                    "rsl-rl-lib",
                    "myo-sim",
                    "numpy",
                    "onnx",
                    "onnxruntime",
                )
            },
            "workflow_sha256": hashlib.sha256(
                Path(__file__).read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest(),
            "synthetic_factors": KNOWN_FACTORS,
            "estimated_factors": config.model.estimate.to_dict()["factors"],
            "capture_quality": preparation["quality"],
            "training": training,
            "cpu_replay": values,
            "native_onnx_action_error_max": action_error,
            "mismatch_rejections": rejected,
            "identity": {
                name: metadata[name]
                for name in (
                    "reference_model_digest",
                    "model_digest",
                    "task_digest",
                    "feature_digest",
                )
            },
        }
        _write(output / "smoke_summary.json", summary)
        return report
    finally:
        target.close()
        nominal.close()
        changed.close()


def main() -> None:
    """Run individual stages or execute the complete pipeline in fresh processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("prepare", "train", "resume", "replay", "all"),
        default="prepare",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("tasks/subject-leg-workflow")
    )
    parser.add_argument("--gpu-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "all":
        prepare(output)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
        for stage in ("train", "resume", "replay"):
            log_path = output / f"{stage}.log"
            print(f"Running {stage}; log: {log_path}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(
                    [
                        str(args.gpu_python),
                        "-m",
                        MODULE,
                        "--stage",
                        stage,
                        "--output",
                        str(output),
                    ],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
    elif args.stage == "prepare":
        print(json.dumps(prepare(output), indent=2))
    elif args.stage in ("train", "resume"):
        print(train(output, resume=args.stage == "resume"))
    else:
        print(json.dumps(replay(output), indent=2))


if __name__ == "__main__":
    main()
