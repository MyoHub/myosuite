#!/usr/bin/env python3
# pylint: disable=no-member
"""Export a registered MyoSuite environment to a loadable MJCF bundle.

The bundle contains:
1. A flattened XML written from ``mujoco.MjSpec.to_xml()``.
2. All mesh/texture/hfield files referenced by the XML, copied under
   ``<output_dir>/assets``.

Example:
    python scripts/export_to_xml.py --env_name myoChallengeBoxingP0-v0
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

import myosuite
from scripts.saber_pose_utils import reset_non_arm_joints_to_upright


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env_name",
        required=True,
        help="Registered Gym/Gymnasium environment name.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("exports"),
        help="Directory where XML bundle will be written.",
    )
    parser.add_argument(
        "--output_xml",
        default="model.xml",
        help="Output XML filename inside output_dir.",
    )
    parser.add_argument(
        "--add-grasp-q0-key",
        action="store_true",
        help="Bake a post-warmup grasp pose keyframe into exported XML.",
    )
    parser.add_argument(
        "--key-name",
        default="grasp_q0",
        help="Keyframe name to use when --add-grasp-q0-key is set.",
    )
    parser.add_argument(
        "--grasp-warmup-steps",
        type=int,
        default=120,
        help="Warmup step count used to build grasp_q0.",
    )
    parser.add_argument(
        "--grasp-action-level",
        type=float,
        default=0.85,
        help="Constant action level in [0,1] used during grasp warmup.",
    )
    return parser.parse_args()


def _as_mjcf_vec(values: np.ndarray) -> str:
    """Format vector for MJCF attribute text."""
    return " ".join(f"{float(v):.12g}" for v in values.tolist())


def _inject_keyframe(
    xml_text: str,
    key_name: str,
    qpos: np.ndarray,
    qvel: np.ndarray | None = None,
) -> str:
    """Insert or replace a keyframe in MJCF text."""
    root = ET.fromstring(xml_text)
    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")

    existing = None
    for node in keyframe.findall("key"):
        if node.attrib.get("name") == key_name:
            existing = node
            break
    if existing is None:
        existing = ET.SubElement(keyframe, "key")
    existing.attrib["name"] = key_name
    existing.attrib["qpos"] = _as_mjcf_vec(qpos)
    if qvel is not None:
        existing.attrib["qvel"] = _as_mjcf_vec(qvel)

    return ET.tostring(root, encoding="unicode")


def _capture_grasp_q0(
    env: gym.Env,
    warmup_steps: int,
    action_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out constant action and capture post-warmup qpos/qvel."""
    env.reset(seed=0)
    unwrapped = env.unwrapped
    model = unwrapped.model
    data = unwrapped.data
    reset_qpos = data.qpos.copy()
    action = np.full(
        env.action_space.shape,
        float(np.clip(action_level, 0.0, 1.0)),
        dtype=np.float32,
    )
    for _ in range(max(0, int(warmup_steps))):
        env.step(action)

    reset_non_arm_joints_to_upright(model, data, reset_qpos)

    snap_fn = getattr(unwrapped, "_snap_free_sabers_to_grasp_sites", None)
    if callable(snap_fn):
        snap_fn()

    mujoco.mj_forward(model, data)
    return data.qpos.copy(), data.qvel.copy()


def _sanitize_name(text: str) -> str:
    """Convert text into filesystem-safe slug.

    Args:
        text: Input string.

    Returns:
        Sanitized string.
    """

    keep = {"-", "_", "."}
    return "".join(ch if ch.isalnum() or ch in keep else "_" for ch in text)


def _extract_xml_text(env: gym.Env) -> str:
    """Extract environment MJCF as XML text.

    Args:
        env: Gym/Gymnasium environment.

    Returns:
        XML text loadable by ``mujoco.MjModel.from_xml_path``.

    Raises:
        RuntimeError: If neither ``MjSpec`` nor ``MjModel`` is available.
    """

    unwrapped = env.unwrapped

    # 1. Direct MjSpec attributes (standard single-agent path and multi-agent path).
    for attr in ("_mj_spec", "mj_spec", "_spec", "obsd_mj_spec"):
        value = getattr(unwrapped, attr, None)
        if isinstance(value, mujoco.MjSpec):
            return value.to_xml()

    # 2. Compiled MjModel — works for XML-loaded models.
    model = getattr(unwrapped, "model", None)
    if model is None:
        model = getattr(unwrapped, "mj_model", None)
    if isinstance(model, mujoco.MjModel):
        try:
            with tempfile.TemporaryDirectory(prefix="myosuite_export_") as tmpdir:
                tmp_xml = Path(tmpdir) / "model.xml"
                mujoco.mj_saveLastXML(str(tmp_xml), model)
                return tmp_xml.read_text(encoding="utf-8")
        except mujoco.FatalError:
            pass

    # 3. Single-agent modular env: rebuild spec from TaskConfig.
    task_config = getattr(unwrapped, "_task_config", None)
    if task_config is not None and hasattr(task_config, "model"):
        from myosuite.core.muscle_conditions import apply_sarcopenia_to_spec
        from myosuite.envs.modular_env import _build_model_from_task_model

        _, fallback_spec = _build_model_from_task_model(task_config.model)
        scene = getattr(task_config, "scene", None)
        if callable(scene):
            fallback_spec = scene(fallback_spec)
        if any(
            getattr(condition, "condition", "") == "sarcopenia"
            for condition in getattr(task_config, "actuators", [])
        ):
            fallback_spec = apply_sarcopenia_to_spec(fallback_spec)
        return fallback_spec.to_xml()

    # 4. Multi-agent modular env: rebuild spec via MultiAgentTaskConfig.build_spec().
    multi_agent_cfg = getattr(unwrapped, "_config", None)
    if multi_agent_cfg is not None and hasattr(multi_agent_cfg, "build_spec"):
        spec = multi_agent_cfg.build_spec()
        if isinstance(spec, mujoco.MjSpec):
            return spec.to_xml()

    raise RuntimeError("Environment does not expose a usable MjSpec/MjModel.")


def _discover_source_xml(env: gym.Env) -> Path | None:
    """Best-effort source XML path discovery.

    Args:
        env: Gym/Gymnasium environment.

    Returns:
        Path to source XML if discoverable, else None.
    """

    unwrapped = env.unwrapped
    for attr in ("_xml_path", "model_path"):
        raw = getattr(unwrapped, attr, None)
        if isinstance(raw, str) and raw.endswith(".xml"):
            candidate = Path(raw).expanduser()
            if candidate.exists():
                return candidate.resolve()
    return None


def _collect_file_nodes(root: ET.Element) -> list[ET.Element]:
    """Collect XML nodes with ``file`` attributes.

    Args:
        root: XML root element.

    Returns:
        List of nodes that reference file assets.
    """

    nodes: list[ET.Element] = []
    for node in root.iter():
        if "file" in node.attrib:
            nodes.append(node)
    return nodes


def _resolve_asset_path(
    file_value: str,
    source_xml: Path | None,
    repo_root: Path,
) -> Path:
    """Resolve an asset path from XML file attribute.

    Args:
        file_value: Value from XML ``file`` attribute.
        source_xml: Source XML path if known.
        repo_root: Repository root.

    Returns:
        Resolved absolute path.

    Raises:
        FileNotFoundError: If asset cannot be resolved.
    """

    raw_path = Path(file_value)
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path.resolve()

    candidates: list[Path] = []
    xml_root = ET.fromstring(
        source_xml.read_text(encoding="utf-8") if source_xml else "<mujoco/>"
    )
    compiler = xml_root.find("compiler")
    meshdir = compiler.attrib.get("meshdir", "") if compiler is not None else ""
    texturedir = compiler.attrib.get("texturedir", "") if compiler is not None else ""

    if source_xml is not None:
        candidates.append((source_xml.parent / raw_path).resolve())
        if meshdir:
            candidates.append((source_xml.parent / meshdir / raw_path).resolve())
        if texturedir:
            candidates.append((source_xml.parent / texturedir / raw_path).resolve())
    candidates.append((repo_root / raw_path).resolve())
    candidates.append((repo_root / "myosuite" / raw_path).resolve())
    # Common MuJoCo asset roots used by MyoSuite specs when compiler meshdir/texturedir
    # context is unavailable in exported XML.
    candidates.append(
        (repo_root / "myosuite" / "simhive" / "myo_model" / raw_path).resolve()
    )
    candidates.append(
        (repo_root / "myosuite" / "simhive" / "myo_sim" / raw_path).resolve()
    )
    candidates.append(
        (repo_root / "myosuite" / "simhive" / "MPL_sim" / raw_path).resolve()
    )
    candidates.append(
        (repo_root / "myosuite" / "simhive" / "YCB_sim" / raw_path).resolve()
    )
    candidates.append(
        (repo_root / "myosuite" / "envs" / "myo" / "assets" / raw_path).resolve()
    )

    def _score(path: Path) -> tuple[int, int]:
        path_str = str(path)
        penalty = 0
        if "site-packages" in path_str or ".venv" in path_str:
            penalty += 100
        if ".venv_install_check" in path_str:
            penalty += 100
        if "/exports/" in path_str:
            penalty += 200
        if str(repo_root / "myosuite") in path_str:
            penalty -= 10
        return penalty, len(path_str)

    existing = [candidate for candidate in candidates if candidate.exists()]
    if existing:
        return sorted(existing, key=_score)[0]

    raise FileNotFoundError(
        f"Could not resolve asset path '{file_value}'. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _copy_assets_and_rewrite_xml(
    xml_text: str,
    output_xml: Path,
    source_xml: Path | None,
    repo_root: Path,
) -> tuple[int, int]:
    """Copy referenced assets and rewrite XML file attributes to local bundle.

    Args:
        xml_text: MJCF XML text.
        output_xml: Destination XML path.
        source_xml: Original source XML if known.
        repo_root: Repository root.

    Returns:
        Tuple ``(assets_referenced, assets_copied)``.
    """

    root = ET.fromstring(xml_text)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.attrib["meshdir"] = "."
    compiler.attrib["texturedir"] = "."
    file_nodes = _collect_file_nodes(root)
    output_assets_root = output_xml.parent / "assets"
    output_assets_root.mkdir(parents=True, exist_ok=True)

    copied_paths: set[Path] = set()
    for node in file_nodes:
        original_value = node.attrib["file"]
        if original_value.startswith("assets/"):
            continue

        resolved = _resolve_asset_path(
            file_value=original_value,
            source_xml=source_xml,
            repo_root=repo_root,
        )
        try:
            rel = resolved.relative_to(repo_root)
        except ValueError:
            rel = Path("external") / resolved.name
        target = output_assets_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        if resolved not in copied_paths:
            shutil.copy2(resolved, target)
            copied_paths.add(resolved)

        node.attrib["file"] = str(Path("assets") / rel).replace("\\", "/")

    tree = ET.ElementTree(root)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    return len(file_nodes), len(copied_paths)


def main() -> None:
    """Export env model as XML + copied assets bundle."""

    args = parse_args()
    repo_root = Path.cwd().resolve()

    myosuite.register_all_envs()

    env = gym.make(args.env_name)
    try:
        xml_text = _extract_xml_text(env)
        source_xml = _discover_source_xml(env)
        if args.add_grasp_q0_key:
            qpos, qvel = _capture_grasp_q0(
                env,
                warmup_steps=args.grasp_warmup_steps,
                action_level=args.grasp_action_level,
            )
            xml_text = _inject_keyframe(
                xml_text,
                key_name=args.key_name,
                qpos=qpos,
                qvel=qvel,
            )
    finally:
        env.close()

    bundle_dir = args.output_dir / _sanitize_name(args.env_name)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    output_xml = bundle_dir / args.output_xml

    referenced, copied = _copy_assets_and_rewrite_xml(
        xml_text=xml_text,
        output_xml=output_xml,
        source_xml=source_xml,
        repo_root=repo_root,
    )

    print("Export complete")
    print(f"- env: {args.env_name}")
    print(f"- xml: {output_xml}")
    print(f"- source_xml: {source_xml if source_xml else 'unknown'}")
    print(f"- file refs in xml: {referenced}")
    print(f"- unique assets copied: {copied}")


if __name__ == "__main__":
    main()
