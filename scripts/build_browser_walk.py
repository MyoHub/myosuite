"""Build a pinned directional-walking policy with mjswan and provenance."""

from __future__ import annotations

import importlib.metadata
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

BASELINE_REVISION = "ecfd7ed0d8b2b3211fe73b14d9538b419970d6df"
ENV_ID = "myoLegDirectionalForward-v0"


@dataclass
class BuildConfig:
    """Keep artifacts outside tracked source and fetch weights at a commit."""

    output: Path = Path("tasks/browser-walk")
    checkpoint: Path | None = None
    serve: bool = False
    port: int = 8787


def browser_actions(cfg: Any, model: mujoco.MjModel) -> dict[str, Any]:
    """Adapt the plain walking excitation pipeline, rejecting unsupported features.

    Args:
        cfg: Actual mjlab task configuration.
        model: CPU model in policy actuator order.

    Returns:
        Native mjswan muscle actions with identical clipping.
    """
    from mjswan.envs.mdp.actions import MuscleActivationActionCfg

    from myosuite.envs.myo.backends.mjlab.tasks.mdp.actions import MyoActionCfg

    if len(cfg.actions) != 1:
        raise ValueError("The walking pilot requires one muscle action term")
    name, action = next(iter(cfg.actions.items()))
    if (
        type(action) is not MyoActionCfg
        or not action.normalize_act
        or tuple(action.action_range) != (0.0, 1.0)
        or action.muscle_sigmoid
        or action.muscle_fatigue
        or action.reroute is not None
        or action.excitation_stages
        or action.clip is not None
        or action.motor_noise.constant_std != 0
        or action.motor_noise.signal_dependent_std != 0
        or not np.all(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)
    ):
        raise ValueError(
            "The walking pilot supports the plain excitation pipeline only"
        )
    return {
        name: MuscleActivationActionCfg(
            action_mode="excitation",
            actuator_names=(f"{re.escape(action.entity_name)}/.*",),
        )
    }


def _policy_parity(env: Any, checkpoint: Path, policy_path: Path) -> dict[str, Any]:
    import onnxruntime as ort

    from myosuite.utils.rslrl_policy import load_rslrl_policy

    policy = load_rslrl_policy(checkpoint, env.action_space.shape[0])
    session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    obs, _ = env.reset(seed=0)
    errors = []
    for _ in range(64):
        action = np.asarray(policy.act(obs)).reshape(env.action_space.shape)
        exported = session.run(None, {session.get_inputs()[0].name: obs[None]})[0][0]
        np.testing.assert_allclose(exported, action, rtol=1e-5, atol=1e-5)
        errors.append(float(np.max(np.abs(exported - action))))
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset(seed=0)
    return {"samples": len(errors), "max_absolute_action_error": max(errors)}


def _runtime_versions() -> dict[str, Any]:
    import mjswan

    modules = Path(mjswan.__file__).parent / "template" / "node_modules"
    versions = {
        name: json.loads((modules / name / "package.json").read_text(encoding="utf-8"))[
            "version"
        ]
        for name in ("mujoco", "onnxruntime-web", "three")
    }
    direct = importlib.metadata.distribution("mjswan").read_text("direct_url.json")
    versions["mjswan_builder_commit"] = (
        json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None
    )
    return versions


def build(config: BuildConfig) -> Path:
    """Build and verify a static bundle without publishing it.

    Args:
        config: Output, checkpoint and loopback server settings.

    Returns:
        Output directory including parity and provenance JSON.
    """
    import mjswan
    import onnx
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from mjlab.tasks.registry import load_env_cfg
    from mjswan.managers.command_manager import CommandBinding

    from myosuite import make_env
    from myosuite.envs.myo.backends.mjlab import tasks  # noqa: F401
    from myosuite.envs.myo.backends.mjlab.tasks import mdp
    from myosuite.utils.deployment_manifest import (
        artifact_record,
        build_deployment_manifest,
        model_identity,
        validate_deployment_manifest,
        write_deployment_manifest,
    )
    from myosuite.utils.export_onnx import export_rslrl_to_onnx

    if mujoco.__version__ != importlib.metadata.version("mujoco"):
        raise RuntimeError(
            "Loaded MuJoCo differs from the installed mujoco distribution"
        )
    output = config.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = None
    checkpoint = config.checkpoint
    if checkpoint is None:
        filename = f"checkpoints/{ENV_ID}/model_583.pt"
        source = {
            "repo_id": "myohub/myosuite-3-baselines",
            "filename": filename,
            "revision": BASELINE_REVISION,
        }
        snapshot = snapshot_download(
            source["repo_id"],
            revision=BASELINE_REVISION,
            allow_patterns=[f"checkpoints/{ENV_ID}/*"],
        )
        checkpoint = Path(snapshot) / filename
    checkpoint = checkpoint.resolve()
    env = make_env(ENV_ID)
    try:
        policy_path = output / "policy.onnx"
        export_rslrl_to_onnx(
            checkpoint,
            policy_path,
            env.observation_space.shape[0],
            env.action_space.shape[0],
        )
        parity = _policy_parity(env, checkpoint, policy_path)
        cfg = load_env_cfg(ENV_ID, play=True)
        actions = browser_actions(cfg, env.unwrapped.model)
        cfg.terminations.pop(mdp.SYNC_TERM)
        mjswan.register_command(
            "HeadingCommandCfg",
            CommandBinding(state_fields=["_target"], command_field="_target"),
        )
        builder = mjswan.Builder(debug=True)
        project = builder.add_project(name="MyoSuite walking pilot")
        scene = project.add_scene_mjlab(ENV_ID, env_cfg=cfg)
        scene.set_viewer(
            mjswan.ViewerConfig(
                distance=3.0, elevation=-12.0, azimuth=45.0, lookat=(0.0, 0.0, 0.2)
            )
        )
        scene.add_policy(
            name="Directional forward baseline",
            policy=onnx.load(policy_path),
            actions=actions,
        )
        app = builder.build(output / "dist")
        dist = output / "dist"
        bundle = json.loads((dist / "manifest.json").read_text(encoding="utf-8"))
        bundle_project = bundle["projects"][0]
        bundle_scene = bundle_project["scenes"][0]
        shipped = (
            dist
            / bundle_project["id"]
            / bundle_scene["id"]
            / bundle_scene["policies"][0]["onnx"]
        )
        parity["shipped_policy"] = _policy_parity(env, checkpoint, shipped)
        for graph in dist.rglob("*.onnx"):
            onnx.checker.check_model(str(graph), full_check=True)
            ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
        actual_model = mujoco.MjSpec.from_zip(
            str(next(dist.rglob("scene.mjz")))
        ).compile()
        terms = [
            {
                "name": "joint_pos",
                "shape": [actual_model.nq],
                "units": "MuJoCo qpos: m, rad, unit quaternion",
            },
            {
                "name": "joint_vel",
                "shape": [actual_model.nv],
                "units": "qvel times control_dt: m, rad",
                "scale": float(env.unwrapped._ctrl_dt),
            },
            {"name": "muscle_act", "shape": [actual_model.na], "units": "1"},
            {"name": "root_planar_vel", "shape": [2], "units": "m/s"},
            {"name": "heading_cmd", "shape": [2], "units": "1"},
        ]
        if [term["name"] for term in terms] != list(cfg.observations["actor"].terms):
            raise ValueError("The walking observation contract changed")
        if sum(term["shape"][0] for term in terms) != env.observation_space.shape[0]:
            raise ValueError("The walking observation dimensions changed")
        interface = {
            "env_id": ENV_ID,
            "observations": terms,
            "history": {"length": 1},
            "actions": [actual_model.actuator(i).name for i in range(actual_model.nu)],
            "action_mapping": {"quantity": "muscle excitation", "clip": [0, 1]},
            "commands": {
                "heading": {
                    "state_fields": ["_target"],
                    "initial": list(cfg.commands["heading"].heading_dir),
                }
            },
            "source_model": model_identity(env.unwrapped.model),
        }
        manifest = build_deployment_manifest(
            actual_model,
            interface=interface,
            control_dt=float(env.unwrapped._ctrl_dt),
            artifacts={
                path.relative_to(dist).as_posix(): path
                for path in sorted(dist.rglob("*"))
                if path.is_file() and path.name != "bundle.manifest.json"
            },
            checkpoint=artifact_record(checkpoint, source=source),
            normalization={"kind": "frozen_in_policy", "source": "checkpoint"},
            backend="browser",
            runtime=_runtime_versions(),
        )
        write_deployment_manifest(dist / "bundle", manifest)
        validate_deployment_manifest(manifest, artifact_root=dist)
        (output / "parity.json").write_text(
            json.dumps(parity, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        env.close()
    if config.serve:
        app.launch(host="127.0.0.1", port=config.port, open_browser=False)
    return output


if __name__ == "__main__":
    import tyro

    print(build(tyro.cli(BuildConfig)))
