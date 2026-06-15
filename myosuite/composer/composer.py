# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""
MyoSuite Web Composer — interactive MJCF scene builder powered by Viser.

Features:
  1. Pre-set MyoSuite scenes  (flat floor, full studio, no-pedestal, no-floor).
  2. MSK model fragments       (hand, arm, elbow, leg, body, torso, …).
  3. Native MuJoCo primitives  (box, sphere, cylinder, capsule, ellipsoid).
  4. Mesh objects               from the bundled asset library *or* from disk.

Every element has a 3-D gizmo for interactive repositioning.
Export writes a self-contained MJCF XML file.

Usage::

    from myosuite.composer.composer import WebComposer
    c = WebComposer(port=8084)
    c.run()          # visit http://localhost:8084
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
import viser

from myosuite.physics.quat_math import euler2quat, quat2euler
from myosuite.utils.simhive_path import get_simhive_asset_root

# ---------------------------------------------------------------------------
# Asset paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent.parent  # myosuite/
_ASSETS = _HERE / "envs" / "myo" / "assets"

# ---------------------------------------------------------------------------
# MuscleMimic model catalogue
# ---------------------------------------------------------------------------


def _build_musclemimic_catalogue() -> dict[str, Path]:
    """Return {model_name: xml_path} for installed musclemimic_models.

    Returns an empty dict when ``musclemimic_models`` is not installed so the
    rest of the composer degrades gracefully.
    """
    try:
        from musclemimic_models import (  # type: ignore[import-untyped]
            REGISTRY,
            get_xml_path,
        )

        return {name: Path(get_xml_path(name)) for name in REGISTRY}
    except ImportError:
        return {}


MUSCLEMIMIC_CATALOGUE: dict[str, Path] = _build_musclemimic_catalogue()

# Human-readable labels for known model keys.
_MIMIC_LABELS: dict[str, str] = {
    "bimanual": "Bimanual Arm (MuscleMimic)",
    "myofullbody": "Full Body (MuscleMimic)",
}

# ---------------------------------------------------------------------------
# Fragment geometry cache
# ---------------------------------------------------------------------------

# Maps fragment XML path → merged (vertices float32, faces uint32) or None on failure.
_FRAGMENT_GEOM_CACHE: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}

# MuJoCo geom groups rendered in the Viser preview.
# Group 0: bone meshes. Group 4: skin capsules.
# Group 3 (muscle-wrapping volumes) is intentionally excluded.
_VISUAL_GROUPS: frozenset[int] = frozenset({0, 4})


def _extract_fragment_geometry(
    fragment_path: Path,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compile *fragment_path* and extract its visual mesh geometry.

    Renders geoms in ``_VISUAL_GROUPS``:
      - **Group 0** — bone meshes (``mjGEOM_MESH``).
      - **Group 4** — skin capsules / ellipsoids (converted to triangle meshes
        via trimesh; skipped gracefully if trimesh is unavailable).

    Results are merged into a single (vertices, faces) pair and cached so
    subsequent adds of the same fragment are free.

    Args:
        fragment_path: Absolute path to the fragment XML file.

    Returns:
        ``(vertices float32 [N,3], faces uint32 [F,3])`` in the fragment's
        local frame (worldbody at origin), or ``None`` if compilation fails.
    """
    key = str(fragment_path)
    if key in _FRAGMENT_GEOM_CACHE:
        return _FRAGMENT_GEOM_CACHE[key]

    try:
        frag_spec = mujoco.MjSpec.from_file(str(fragment_path))
        model = frag_spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
    except Exception as exc:
        print(f"[WebComposer] Fragment compile failed ({fragment_path.name}): {exc}")
        _FRAGMENT_GEOM_CACHE[key] = None
        return None

    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vert_offset = 0

    for i in range(model.ngeom):
        # Skip geoms parented to worldbody: they are scene decorations
        # (floor plane, room mesh) included via the fragment's scene XML,
        # not part of the musculoskeletal model itself.
        if int(model.geom_bodyid[i]) == 0:
            continue

        group = int(model.geom_group[i])
        if group not in _VISUAL_GROUPS:
            continue
        if model.geom_rgba[i, 3] < 0.01:
            continue

        gtype = int(model.geom_type[i])
        pos = data.geom_xpos[i].copy()
        mat = data.geom_xmat[i].copy().reshape(3, 3)
        size = model.geom_size[i].copy()

        verts: np.ndarray | None = None
        faces: np.ndarray | None = None

        if gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(model.geom_dataid[i])
            if mid < 0:
                continue
            va = int(model.mesh_vertadr[mid])
            vn = int(model.mesh_vertnum[mid])
            fa = int(model.mesh_faceadr[mid])
            fn = int(model.mesh_facenum[mid])
            local_v = model.mesh_vert[va : va + vn].copy()
            verts = (mat @ local_v.T).T + pos
            faces = model.mesh_face[fa : fa + fn].copy()
        else:
            # Approximate primitives (skin capsules, ellipsoids) via trimesh.
            try:
                import trimesh  # type: ignore[import-untyped]

                tm = _primitive_to_trimesh(trimesh, gtype, size)
                if tm is not None:
                    verts = (mat @ np.asarray(tm.vertices, dtype=float).T).T + pos
                    faces = np.asarray(tm.faces)
            except Exception:
                pass

        if verts is not None and faces is not None and len(verts) > 0:
            all_verts.append(verts.astype(np.float32))
            all_faces.append(faces.astype(np.uint32) + vert_offset)
            vert_offset += len(verts)

    if not all_verts:
        _FRAGMENT_GEOM_CACHE[key] = None
        return None

    result: tuple[np.ndarray, np.ndarray] = (
        np.concatenate(all_verts, axis=0),
        np.concatenate(all_faces, axis=0),
    )
    _FRAGMENT_GEOM_CACHE[key] = result
    return result


def _primitive_to_trimesh(trimesh_mod: Any, gtype: int, size: np.ndarray) -> Any:
    """Convert a MuJoCo primitive geom to a low-res trimesh for preview.

    Args:
        trimesh_mod: The imported trimesh module.
        gtype: ``mujoco.mjtGeom`` integer value.
        size: MuJoCo geom size array.

    Returns:
        A trimesh ``Trimesh`` object, or ``None`` for unsupported types.
    """
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh_mod.creation.icosphere(subdivisions=2, radius=float(size[0]))
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh_mod.creation.box(
            extents=[float(size[0]) * 2, float(size[1]) * 2, float(size[2]) * 2]
        )
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return trimesh_mod.creation.capsule(
            radius=float(size[0]), height=float(size[1]) * 2, count=[8, 8]
        )
    if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return trimesh_mod.creation.cylinder(
            radius=float(size[0]), height=float(size[1]) * 2, sections=16
        )
    if gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        m = trimesh_mod.creation.icosphere(subdivisions=2, radius=1.0)
        m.vertices *= size[:3]
        return m
    return None


def _load_musclemimic_spec(xml_path: Path) -> mujoco.MjSpec:
    """Load a MuscleMimic XML with all scene includes removed.

    MuscleMimic model XMLs include a scene file (floor, lights, branding) that
    must be stripped before attaching the model into the composer spec so that
    the composer's own scene setup is not overridden.

    The approach mirrors ``bimanual_model._load_spec_with_default_scene`` but
    drops the scene include entirely rather than substituting it.

    Args:
        xml_path: Absolute path to the MuscleMimic entry XML.

    Returns:
        A :class:`mujoco.MjSpec` containing only the body model (no scene).
    """
    import tempfile

    lines = xml_path.read_text(encoding="utf-8").splitlines()
    stripped = [
        line
        for line in lines
        if not ("<include" in line.lower() and "scene" in line.lower())
    ]
    with tempfile.NamedTemporaryFile(
        "w",
        dir=xml_path.parent,
        suffix=".xml",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write("\n".join(stripped))
        tmp_path = Path(tmp.name)
    try:
        return mujoco.MjSpec.from_file(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


def _quat_to_euler_deg(quat: np.ndarray) -> tuple[float, float, float]:
    """Convert a (w, x, y, z) unit quaternion to intrinsic XYZ Euler angles (degrees).

    Args:
        quat: Unit quaternion as ``[w, x, y, z]``.

    Returns:
        ``(roll_deg, pitch_deg, yaw_deg)`` in the range ``[-180, 180]``.
    """
    euler_rad = quat2euler(np.asarray(quat, dtype=np.float64))
    roll, pitch, yaw = np.rad2deg(euler_rad)
    return float(roll), float(pitch), float(yaw)


def _euler_deg_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert intrinsic XYZ Euler angles (degrees) to a (w, x, y, z) unit quaternion.

    Args:
        roll: Rotation around X axis in degrees.
        pitch: Rotation around Y axis in degrees.
        yaw: Rotation around Z axis in degrees.

    Returns:
        Unit quaternion as ``np.ndarray([w, x, y, z])``.
    """
    euler_rad = np.deg2rad(np.asarray([roll, pitch, yaw], dtype=np.float64))
    return euler2quat(euler_rad).astype(np.float64)


# ---------------------------------------------------------------------------
# Scene presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenePreset:
    """Description of a named MyoSuite scene preset.

    Args:
        name: Machine-readable identifier.
        label: Human-readable label shown in the UI.
        has_floor: Whether the scene includes a floor plane.
        has_pedestal: Whether the scene includes a cylindrical pedestal.
        floor_z: Z position of the floor plane centre.
    """

    name: str
    label: str
    has_floor: bool = True
    has_pedestal: bool = True
    floor_z: float = -0.4


SCENE_PRESETS: dict[str, ScenePreset] = {
    "flat_floor": ScenePreset(
        "flat_floor",
        "Flat Floor",
        has_floor=True,
        has_pedestal=False,
        floor_z=0.0,
    ),
    "myosuite_scene": ScenePreset(
        "myosuite_scene",
        "MyoSuite Studio (with Pedestal)",
        has_floor=True,
        has_pedestal=True,
        floor_z=-0.4,
    ),
    "myosuite_scene_noPedestal": ScenePreset(
        "myosuite_scene_noPedestal",
        "MyoSuite Studio (no Pedestal)",
        has_floor=True,
        has_pedestal=False,
        floor_z=0.0,
    ),
    "no_floor": ScenePreset(
        "no_floor",
        "No Floor",
        has_floor=False,
        has_pedestal=False,
        floor_z=0.0,
    ),
}

# ---------------------------------------------------------------------------
# Fragment catalogue
# ---------------------------------------------------------------------------

FRAGMENT_CATALOGUE: dict[str, str] = {
    "hand": "hand/myohand.xml",
    "elbow": "elbow/myoelbow_2dof6muscles.xml",
    "arm": "arm/myoarm.xml",
    "finger": "finger/myofinger_v0.xml",
    "leg": "leg/myolegs.xml",
    "body": "body/myobody.xml",
    "torso": "torso/myotorso.xml",
    "osl": "osl/myolegs_osl.xml",
}

# Approximate visual half-extents (x, y, z) for bounding-box placeholders.
_FRAGMENT_BOUNDS: dict[str, tuple[float, float, float]] = {
    "hand": (0.10, 0.08, 0.15),
    "elbow": (0.08, 0.15, 0.10),
    "arm": (0.15, 0.15, 0.40),
    "finger": (0.02, 0.02, 0.06),
    "leg": (0.20, 0.15, 0.55),
    "body": (0.30, 0.20, 0.90),
    "torso": (0.25, 0.18, 0.40),
    "osl": (0.10, 0.10, 0.35),
}

# ---------------------------------------------------------------------------
# Primitive catalogue
# ---------------------------------------------------------------------------

PRIMITIVE_TYPES: list[str] = ["box", "sphere", "cylinder", "capsule", "ellipsoid"]

_GEOM_TYPE_MAP: dict[str, int] = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
}

# ---------------------------------------------------------------------------
# Bundled mesh asset library
# ---------------------------------------------------------------------------

MESH_ASSETS: dict[str, Path] = {
    p.name: p
    for p in [
        _ASSETS / "PunchingBag.obj",
        _ASSETS / "paddle.obj",
        _ASSETS / "tabletennis_table.obj",
        _ASSETS / "tabletennis_net.obj",
    ]
    if p.exists()
}

# ---------------------------------------------------------------------------
# Scene element record
# ---------------------------------------------------------------------------


@dataclass
class SceneElement:
    """One item placed in the composed scene.

    Args:
        eid: Unique element ID (used as Viser path prefix and MJCF name prefix).
        kind: Element category.
        label: Display label in the sidebar.
        pos: World-frame position (x, y, z).
        quat: Orientation quaternion (w, x, y, z) — MJCF convention.
        fragment_name: Fragment key from ``FRAGMENT_CATALOGUE`` (fragment only).
        primitive_type: One of ``PRIMITIVE_TYPES`` (primitive only).
        geom_size: Half-extents or radius(es) passed to MuJoCo.
        rgba: RGBA colour [0..1].
        mass: Optional explicit mass in kg.
        mesh_path: Path to a ``.obj`` / ``.stl`` mesh file (mesh only).
        texture_path: Optional texture path (mesh only).
    """

    eid: str
    kind: Literal["fragment", "primitive", "mesh", "mimic"]
    label: str
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    fragment_name: str | None = None
    primitive_type: str | None = None
    geom_size: np.ndarray | None = None
    rgba: np.ndarray | None = None
    mass: float | None = None
    mesh_path: Path | None = None
    texture_path: Path | None = None
    # MuscleMimic models: absolute path to the entry XML.
    mimic_xml_path: Path | None = None


# ---------------------------------------------------------------------------
# WebComposer
# ---------------------------------------------------------------------------


class WebComposer:
    """Interactive MJCF scene composer served via a Viser browser UI.

    Args:
        port: Port for the Viser web server (default 8084).
        export_dir: Default directory for MJCF exports (default: cwd).
    """

    def __init__(self, port: int = 8084, export_dir: Path | None = None) -> None:
        self._elements: dict[str, SceneElement] = {}
        self._counter = 0
        self._lock = threading.Lock()
        self._active_scene = "flat_floor"

        # Ordered list of eids for Ctrl+Z undo (most-recent last).
        self._undo_stack: list[str] = []

        self._server = viser.ViserServer(port=port)
        self._server.scene.set_up_direction("+z")
        self._server.scene.configure_default_lights()

        # Viser handles for scene background (floor, pedestal, grid)
        self._scene_handles: list[Any] = []

        # Per-element Viser handles: eid → list
        self._element_visuals: dict[str, list[Any]] = {}
        self._element_gizmos: dict[str, Any] = {}

        # Per-element GUI folder handles
        self._element_gui_folders: dict[str, Any] = {}

        self._export_dir = export_dir or Path.cwd()
        self._export_path_handle: Any = None

        self._build_gui()
        self._apply_scene("flat_floor")

    # ------------------------------------------------------------------
    # GUI construction
    # ------------------------------------------------------------------

    def _build_gui(self) -> None:
        """Create the full sidebar GUI."""
        gui = self._server.gui
        gui.set_panel_label("MyoSuite Composer")

        tabs = gui.add_tab_group()

        with tabs.add_tab("Scene", viser.Icon.MAP):
            self._build_scene_tab(gui)

        with tabs.add_tab("MSK", viser.Icon.BONE):
            self._build_msk_tab(gui)

        with tabs.add_tab("Primitives", viser.Icon.BOX):
            self._build_primitives_tab(gui)

        with tabs.add_tab("Meshes", viser.Icon.CIRCLE):
            self._build_mesh_tab(gui)

        with gui.add_folder("Elements"):
            self._elements_folder_root = gui.add_folder("List")

        with gui.add_folder("Export"):
            self._export_path_handle = gui.add_text(
                "Output path",
                initial_value=str(self._export_dir / "composed_scene.xml"),
            )
            export_btn = gui.add_button("Export MJCF", icon=viser.Icon.DEVICE_FLOPPY)
            undo_btn = gui.add_button(
                "Undo Last  (⌘Z / Ctrl+Z)", icon=viser.Icon.ARROW_BACK_UP
            )
            clear_btn = gui.add_button("Clear All", icon=viser.Icon.TRASH)

        @export_btn.on_click
        def _(_) -> None:
            self._do_export()

        @undo_btn.on_click
        def _(_) -> None:
            self._undo_last()

        @clear_btn.on_click
        def _(_) -> None:
            self._clear_all()

    def _build_scene_tab(self, gui: viser.GuiApi) -> None:
        scene_names = list(SCENE_PRESETS.keys())
        scene_labels = [SCENE_PRESETS[n].label for n in scene_names]
        scene_dd = gui.add_dropdown(
            "Preset", options=scene_labels, initial_value=scene_labels[0]
        )
        apply_btn = gui.add_button("Apply Scene", icon=viser.Icon.REFRESH)

        @apply_btn.on_click
        def _(_) -> None:
            idx = scene_labels.index(scene_dd.value)
            self._apply_scene(scene_names[idx])

    def _build_msk_tab(self, gui: viser.GuiApi) -> None:
        with gui.add_folder("MyoSuite Fragments"):
            frag_names = list(FRAGMENT_CATALOGUE.keys())
            frag_dd = gui.add_dropdown(
                "Model", options=frag_names, initial_value=frag_names[0]
            )
            pos_in = gui.add_vector3(
                "Position (m)", initial_value=(0.0, 0.0, 0.0), step=0.05
            )
            add_btn = gui.add_button("Add Fragment", icon=viser.Icon.PLUS)

            @add_btn.on_click
            def _(_) -> None:
                self._add_fragment(
                    name=frag_dd.value,
                    pos=np.array(pos_in.value),
                )

        with gui.add_folder("MuscleMimic Models"):
            if MUSCLEMIMIC_CATALOGUE:
                mimic_names = list(MUSCLEMIMIC_CATALOGUE.keys())
                mimic_labels = [_MIMIC_LABELS.get(n, n) for n in mimic_names]
                mimic_dd = gui.add_dropdown(
                    "Model", options=mimic_labels, initial_value=mimic_labels[0]
                )
                mimic_pos = gui.add_vector3(
                    "Position (m)", initial_value=(0.0, 0.0, 0.0), step=0.05
                )
                add_mimic_btn = gui.add_button("Add MuscleMimic", icon=viser.Icon.PLUS)

                @add_mimic_btn.on_click
                def _(_) -> None:
                    idx = mimic_labels.index(mimic_dd.value)
                    self._add_mimic(
                        name=mimic_names[idx],
                        pos=np.array(mimic_pos.value),
                    )

            else:
                gui.add_markdown(
                    "_`musclemimic_models` not installed._\n\n"
                    "Install with:\n```\npip install 'myosuite[musclemimic]'\n```"
                )

    def _build_primitives_tab(self, gui: viser.GuiApi) -> None:
        type_dd = gui.add_dropdown("Type", options=PRIMITIVE_TYPES, initial_value="box")
        size_in = gui.add_vector3(
            "Half-size / Radii (m)",
            initial_value=(0.05, 0.05, 0.05),
            step=0.01,
            min=(0.001, 0.001, 0.001),
        )
        color_in = gui.add_rgb("Colour", initial_value=(180, 60, 60))
        alpha_in = gui.add_slider(
            "Alpha", min=0.1, max=1.0, step=0.05, initial_value=1.0
        )
        mass_in = gui.add_number(
            "Mass kg (0 = auto)", initial_value=0.0, min=0.0, step=0.01
        )
        pos_in = gui.add_vector3(
            "Position (m)", initial_value=(0.0, 0.0, 0.3), step=0.05
        )
        add_btn = gui.add_button("Add Primitive", icon=viser.Icon.PLUS)

        @add_btn.on_click
        def _(_) -> None:
            r, g, b = color_in.value
            rgba = np.array([r / 255.0, g / 255.0, b / 255.0, alpha_in.value])
            mass = float(mass_in.value) if mass_in.value > 0.0 else None
            self._add_primitive(
                ptype=type_dd.value,
                geom_size=np.array(size_in.value),
                rgba=rgba,
                pos=np.array(pos_in.value),
                mass=mass,
            )

    def _build_mesh_tab(self, gui: viser.GuiApi) -> None:
        asset_names = list(MESH_ASSETS.keys())

        with gui.add_folder("Asset Library"):
            if asset_names:
                asset_dd = gui.add_dropdown(
                    "Asset", options=asset_names, initial_value=asset_names[0]
                )
                scale_in = gui.add_number(
                    "Scale", initial_value=1.0, min=0.001, step=0.1
                )
                asset_pos = gui.add_vector3(
                    "Position (m)", initial_value=(0.0, 0.0, 0.0), step=0.05
                )
                add_asset_btn = gui.add_button("Add from Library", icon=viser.Icon.PLUS)

                @add_asset_btn.on_click
                def _(_) -> None:
                    self._add_mesh(
                        mesh_path=MESH_ASSETS[asset_dd.value],
                        pos=np.array(asset_pos.value),
                        scale=float(scale_in.value),
                    )

            else:
                gui.add_markdown("_No bundled mesh assets found._")

        with gui.add_folder("Upload from Disk"):
            upload_btn = gui.add_upload_button(
                "Upload .obj / .stl",
                mime_type=".obj,.stl",
                icon=viser.Icon.UPLOAD,
            )
            upload_pos = gui.add_vector3(
                "Position (m)", initial_value=(0.0, 0.0, 0.0), step=0.05
            )

            @upload_btn.on_upload
            def _(event: viser.GuiUploadedFile) -> None:
                tmp = Path(f"/tmp/_myo_composer_{self._counter}_{event.name}")
                tmp.write_bytes(event.content)
                self._add_mesh(mesh_path=tmp, pos=np.array(upload_pos.value))

    # ------------------------------------------------------------------
    # Scene preset
    # ------------------------------------------------------------------

    def _apply_scene(self, name: str) -> None:
        """Switch to a named scene preset and update Viser visuals.

        Args:
            name: Key from ``SCENE_PRESETS``.
        """
        if name not in SCENE_PRESETS:
            return

        for handle in self._scene_handles:
            handle.remove()
        self._scene_handles.clear()

        preset = SCENE_PRESETS[name]
        self._active_scene = name

        grid = self._server.scene.add_grid(
            "/scene/grid", width=10, height=10, cell_size=0.5, plane="xy"
        )
        self._scene_handles.append(grid)

        if preset.has_floor:
            floor = self._server.scene.add_box(
                "/scene/floor",
                color=(180, 180, 180),
                dimensions=(10.0, 10.0, 0.02),
                position=(0.0, 0.0, float(preset.floor_z)),
                opacity=0.6,
            )
            self._scene_handles.append(floor)

        if preset.has_pedestal:
            ped = self._server.scene.add_cylinder(
                "/scene/pedestal",
                radius=1.05,
                height=0.41,
                color=(120, 120, 120),
                position=(0.0, 0.0, float(preset.floor_z) + 0.205),
                opacity=0.5,
            )
            self._scene_handles.append(ped)

        lbl = self._server.scene.add_label(
            "/scene/label",
            text=f"Scene: {preset.label}",
            position=(0.0, 0.0, 0.05),
        )
        self._scene_handles.append(lbl)

    # ------------------------------------------------------------------
    # Adding elements
    # ------------------------------------------------------------------

    def _next_eid(self, kind: str) -> str:
        eid = f"{kind}_{self._counter:04d}"
        self._counter += 1
        return eid

    def _add_fragment(self, name: str, pos: np.ndarray) -> None:
        """Add an MSK fragment placeholder to the scene.

        Args:
            name: Fragment key from ``FRAGMENT_CATALOGUE``.
            pos: Initial world-frame position.
        """
        if name not in FRAGMENT_CATALOGUE:
            return

        fragment_path = get_simhive_asset_root("myo_sim") / FRAGMENT_CATALOGUE[name]
        if not fragment_path.exists():
            print(
                f"[WebComposer] Fragment '{name}' not found at {fragment_path}. "
                "Run: git submodule update --init --recursive"
            )
            return

        with self._lock:
            eid = self._next_eid("frag")
            elem = SceneElement(
                eid=eid,
                kind="fragment",
                label=f"{name} (fragment)",
                pos=pos.copy(),
                fragment_name=name,
            )
            self._elements[eid] = elem
            self._undo_stack.append(eid)

        self._add_fragment_to_viser(elem)
        self._add_element_to_gui(elem)

    def _add_primitive(
        self,
        ptype: str,
        geom_size: np.ndarray,
        rgba: np.ndarray,
        pos: np.ndarray,
        mass: float | None = None,
    ) -> None:
        """Add a native MuJoCo primitive object.

        Args:
            ptype: One of ``PRIMITIVE_TYPES``.
            geom_size: Half-extents or radius(es) in metres.
            rgba: RGBA colour, each channel in [0, 1].
            pos: Initial world-frame position.
            mass: Explicit mass in kg (None = auto from density).
        """
        with self._lock:
            eid = self._next_eid(ptype[:4])
            elem = SceneElement(
                eid=eid,
                kind="primitive",
                label=f"{ptype} {eid.split('_')[-1]}",
                pos=pos.copy(),
                primitive_type=ptype,
                geom_size=geom_size.copy(),
                rgba=rgba.copy(),
                mass=mass,
            )
            self._elements[eid] = elem
            self._undo_stack.append(eid)

        self._add_primitive_to_viser(elem)
        self._add_element_to_gui(elem)

    def _add_mesh(
        self,
        mesh_path: Path,
        pos: np.ndarray,
        scale: float = 1.0,
        texture_path: Path | None = None,
    ) -> None:
        """Add a mesh object from a file path.

        Args:
            mesh_path: Path to an ``.obj`` or ``.stl`` file.
            pos: Initial world-frame position.
            scale: Uniform scale factor.
            texture_path: Optional texture image path.
        """
        if not mesh_path.exists():
            print(f"[WebComposer] Mesh file not found: {mesh_path}")
            return

        with self._lock:
            eid = self._next_eid("mesh")
            elem = SceneElement(
                eid=eid,
                kind="mesh",
                label=f"{mesh_path.stem} (mesh)",
                pos=pos.copy(),
                mesh_path=mesh_path,
                texture_path=texture_path,
                rgba=np.array([0.8, 0.8, 0.8, 1.0]),
                geom_size=np.array([scale, scale, scale]),
            )
            self._elements[eid] = elem
            self._undo_stack.append(eid)

        self._add_mesh_to_viser(elem)
        self._add_element_to_gui(elem)

    def _add_mimic(self, name: str, pos: np.ndarray) -> None:
        """Add a MuscleMimic model (bimanual arm or full body) to the scene.

        Args:
            name: Key from ``MUSCLEMIMIC_CATALOGUE`` (e.g. ``"bimanual"``).
            pos: Initial world-frame position.
        """
        if name not in MUSCLEMIMIC_CATALOGUE:
            print(f"[WebComposer] Unknown MuscleMimic model: {name!r}")
            return

        xml_path = MUSCLEMIMIC_CATALOGUE[name]
        if not xml_path.exists():
            print(f"[WebComposer] MuscleMimic XML not found: {xml_path}")
            return

        label = _MIMIC_LABELS.get(name, f"{name} (mimic)")
        with self._lock:
            eid = self._next_eid("mimic")
            elem = SceneElement(
                eid=eid,
                kind="mimic",
                label=label,
                pos=pos.copy(),
                mimic_xml_path=xml_path,
            )
            self._elements[eid] = elem
            self._undo_stack.append(eid)

        self._add_mimic_to_viser(elem)
        self._add_element_to_gui(elem)

    # ------------------------------------------------------------------
    # Viser visualisation helpers
    # ------------------------------------------------------------------

    def _make_gizmo(self, eid: str, pos: np.ndarray, quat: np.ndarray) -> Any:
        """Create a 3-D transform-controls gizmo at *pos*/*quat*.

        Args:
            eid: Element ID used as the Viser path prefix.
            pos: Initial position.
            quat: Initial orientation (w, x, y, z).

        Returns:
            The Viser ``TransformControlsHandle``.
        """
        gizmo = self._server.scene.add_transform_controls(
            f"/elements/{eid}/gizmo",
            scale=0.5,
            position=tuple(pos.tolist()),
            wxyz=tuple(quat.tolist()),
        )
        self._element_gizmos[eid] = gizmo
        return gizmo

    def _wire_gizmo_transparency(self, eid: str) -> None:
        """Attach drag-start / drag-end callbacks that toggle visual transparency.

        While a gizmo is being dragged the element's visual mesh(es) are hidden
        so the transform handles (translation arrows, rotation rings) are fully
        unobstructed.  Visibility is restored as soon as the drag ends.

        Must be called *after* both ``_make_gizmo`` and the visual handles have
        been stored in ``self._element_visuals[eid]``.

        Args:
            eid: Element ID whose gizmo and visuals to wire up.
        """
        gizmo = self._element_gizmos.get(eid)
        if gizmo is None:
            return

        def _on_drag_start(_: Any) -> None:
            for handle in self._element_visuals.get(eid, []):
                handle.visible = False

        def _on_drag_end(_: Any) -> None:
            for handle in self._element_visuals.get(eid, []):
                handle.visible = True

        gizmo.on_drag_start(_on_drag_start)
        gizmo.on_drag_end(_on_drag_end)

    def _add_fragment_to_viser(self, elem: SceneElement) -> None:
        """Render the fragment's actual bone-mesh geometry in Viser.

        Compiles the fragment XML (cached after first use), extracts visual
        mesh geoms (groups 0 and 4), merges them into a single triangle mesh,
        and adds it as a child of the element's gizmo so interactive
        repositioning works automatically.

        Falls back to a wireframe bounding box if geometry extraction fails.
        """
        self._make_gizmo(elem.eid, elem.pos, elem.quat)

        frag_name = elem.fragment_name or ""
        frag_path = get_simhive_asset_root("myo_sim") / FRAGMENT_CATALOGUE.get(
            frag_name, ""
        )
        geom = _extract_fragment_geometry(frag_path) if frag_path.exists() else None

        if geom is not None:
            vertices, faces = geom
            visual: Any = self._server.scene.add_mesh_simple(
                f"/elements/{elem.eid}/gizmo/visual",
                vertices=vertices,
                faces=faces,
                color=(200, 195, 185),
                opacity=0.5,
                flat_shading=False,
            )
        else:
            # Fallback: wireframe bounding box
            bounds = _FRAGMENT_BOUNDS.get(frag_name, (0.15, 0.15, 0.40))
            visual = self._server.scene.add_box(
                f"/elements/{elem.eid}/gizmo/visual",
                color=(100, 180, 255),
                dimensions=(bounds[0] * 2, bounds[1] * 2, bounds[2] * 2),
                wireframe=True,
                opacity=0.6,
            )

        lbl = self._server.scene.add_label(
            f"/elements/{elem.eid}/gizmo/label",
            text=frag_name,
            position=(0.0, 0.0, _FRAGMENT_BOUNDS.get(frag_name, (0, 0, 0.4))[2] + 0.05),
        )
        self._element_visuals[elem.eid] = [visual, lbl]
        self._wire_gizmo_transparency(elem.eid)

    def _add_primitive_to_viser(self, elem: SceneElement) -> None:
        """Render the appropriate Viser primitive shape for *elem*."""
        self._make_gizmo(elem.eid, elem.pos, elem.quat)

        rgba = elem.rgba if elem.rgba is not None else np.array([0.8, 0.2, 0.2, 1.0])
        color_rgb = tuple(int(c * 255) for c in rgba[:3])
        alpha = float(rgba[3])
        sz = (
            elem.geom_size
            if elem.geom_size is not None
            else np.array([0.05, 0.05, 0.05])
        )
        ptype = elem.primitive_type or "box"

        base_path = f"/elements/{elem.eid}/gizmo/visual"

        if ptype == "box":
            visual: Any = self._server.scene.add_box(
                base_path,
                color=color_rgb,
                dimensions=(float(sz[0]) * 2, float(sz[1]) * 2, float(sz[2]) * 2),
                opacity=alpha,
            )
        elif ptype == "sphere":
            visual = self._server.scene.add_icosphere(
                base_path,
                radius=float(sz[0]),
                color=color_rgb,
                opacity=alpha,
            )
        elif ptype in ("cylinder", "capsule"):
            visual = self._server.scene.add_cylinder(
                base_path,
                radius=float(sz[0]),
                height=float(sz[1]) * 2,
                color=color_rgb,
                opacity=alpha,
            )
        elif ptype == "ellipsoid":
            # Ellipsoid: unit icosphere + non-uniform scale
            visual = self._server.scene.add_icosphere(
                base_path,
                radius=1.0,
                color=color_rgb,
                scale=(float(sz[0]), float(sz[1]), float(sz[2])),
                opacity=alpha,
            )
        else:
            visual = self._server.scene.add_box(
                base_path,
                color=color_rgb,
                dimensions=(0.1, 0.1, 0.1),
                opacity=alpha,
            )

        self._element_visuals[elem.eid] = [visual]
        self._wire_gizmo_transparency(elem.eid)

    def _add_mesh_to_viser(self, elem: SceneElement) -> None:
        """Load the mesh file and render it in Viser.

        Falls back to a wireframe box if *trimesh* is unavailable or
        parsing fails.
        """
        self._make_gizmo(elem.eid, elem.pos, elem.quat)

        scale = float(elem.geom_size[0]) if elem.geom_size is not None else 1.0
        base_path = f"/elements/{elem.eid}/gizmo/visual"

        try:
            import trimesh  # type: ignore[import-untyped]

            loaded = trimesh.load(str(elem.mesh_path), force="mesh")
            vertices = np.array(loaded.vertices, dtype=np.float32) * scale
            faces = np.array(loaded.faces, dtype=np.uint32)
            visual: Any = self._server.scene.add_mesh_simple(
                base_path,
                vertices=vertices,
                faces=faces,
                color=(200, 200, 200),
                opacity=0.5,
                flat_shading=False,
            )
        except Exception as exc:
            print(f"[WebComposer] Could not render mesh '{elem.mesh_path}': {exc}")
            visual = self._server.scene.add_box(
                base_path,
                color=(200, 200, 200),
                dimensions=(0.2, 0.2, 0.2),
                wireframe=True,
            )

        lbl = self._server.scene.add_label(
            f"/elements/{elem.eid}/gizmo/label",
            text=elem.mesh_path.stem if elem.mesh_path else "mesh",
            position=(0.0, 0.0, 0.15),
        )
        self._element_visuals[elem.eid] = [visual, lbl]
        self._wire_gizmo_transparency(elem.eid)

    def _add_mimic_to_viser(self, elem: SceneElement) -> None:
        """Render a MuscleMimic model's bone-mesh geometry in Viser.

        Reuses :func:`_extract_fragment_geometry` which already filters out
        worldbody scene geoms (floor, lights) via ``geom_bodyid != 0``.
        Falls back to a wireframe bounding box if extraction fails.
        """
        self._make_gizmo(elem.eid, elem.pos, elem.quat)

        geom = (
            _extract_fragment_geometry(elem.mimic_xml_path)
            if elem.mimic_xml_path is not None
            else None
        )

        if geom is not None:
            vertices, faces = geom
            visual: Any = self._server.scene.add_mesh_simple(
                f"/elements/{elem.eid}/gizmo/visual",
                vertices=vertices,
                faces=faces,
                color=(185, 200, 210),  # slightly cooler tint to distinguish from myo
                opacity=0.5,
                flat_shading=False,
            )
        else:
            visual = self._server.scene.add_box(
                f"/elements/{elem.eid}/gizmo/visual",
                color=(100, 200, 180),
                dimensions=(0.5, 0.4, 1.8),
                wireframe=True,
                opacity=0.6,
            )

        label_text = _MIMIC_LABELS.get(
            next(
                (
                    k
                    for k, v in MUSCLEMIMIC_CATALOGUE.items()
                    if v == elem.mimic_xml_path
                ),
                "",
            ),
            elem.label,
        )
        lbl = self._server.scene.add_label(
            f"/elements/{elem.eid}/gizmo/label",
            text=label_text,
            position=(0.0, 0.0, 1.85),
        )
        self._element_visuals[elem.eid] = [visual, lbl]
        self._wire_gizmo_transparency(elem.eid)

    # ------------------------------------------------------------------
    # Elements GUI panel
    # ------------------------------------------------------------------

    def _add_element_to_gui(self, elem: SceneElement) -> None:
        """Add transform controls and a Remove button for *elem* in the sidebar.

        Each element gets:
        - A **Position** vector3 input (step 0.01 m) that drives the gizmo.
        - **Roll / Pitch / Yaw** number inputs (step 1°) for rotation.
        - A **Remove** button.

        The controls stay in sync with the 3-D gizmo: dragging the gizmo
        updates the sidebar numbers; changing the sidebar numbers moves the
        gizmo.

        Args:
            elem: The newly added scene element.
        """
        eid = elem.eid
        roll0, pitch0, yaw0 = _quat_to_euler_deg(elem.quat)

        with self._elements_folder_root:
            folder = self._server.gui.add_folder(elem.label, expand_by_default=False)

        with folder:
            pos_handle = self._server.gui.add_vector3(
                "Position (m)",
                initial_value=(
                    float(elem.pos[0]),
                    float(elem.pos[1]),
                    float(elem.pos[2]),
                ),
                step=0.01,
            )
            with self._server.gui.add_folder("Rotation", expand_by_default=False):
                roll_handle = self._server.gui.add_number(
                    "Roll °",
                    initial_value=round(roll0, 1),
                    step=1.0,
                    min=-180.0,
                    max=180.0,
                )
                pitch_handle = self._server.gui.add_number(
                    "Pitch °",
                    initial_value=round(pitch0, 1),
                    step=1.0,
                    min=-180.0,
                    max=180.0,
                )
                yaw_handle = self._server.gui.add_number(
                    "Yaw °",
                    initial_value=round(yaw0, 1),
                    step=1.0,
                    min=-180.0,
                    max=180.0,
                )
            remove_btn = self._server.gui.add_button("Remove", icon=viser.Icon.TRASH)

        # --- GUI → gizmo sync ---
        def _gui_changed(_: Any) -> None:
            pos = np.array(pos_handle.value, dtype=float)
            quat = _euler_deg_to_quat(
                float(roll_handle.value),
                float(pitch_handle.value),
                float(yaw_handle.value),
            )
            with self._lock:
                if eid in self._elements:
                    self._elements[eid].pos = pos
                    self._elements[eid].quat = quat
            gizmo = self._element_gizmos.get(eid)
            if gizmo is not None:
                gizmo.position = tuple(pos.tolist())
                gizmo.wxyz = tuple(quat.tolist())

        pos_handle.on_update(_gui_changed)
        roll_handle.on_update(_gui_changed)
        pitch_handle.on_update(_gui_changed)
        yaw_handle.on_update(_gui_changed)

        # --- gizmo → GUI sync ---
        gizmo = self._element_gizmos.get(eid)
        if gizmo is not None:

            def _gizmo_moved(event: Any) -> None:
                pos = np.array(event.target.position, dtype=float)
                quat = np.array(event.target.wxyz, dtype=float)
                with self._lock:
                    if eid in self._elements:
                        self._elements[eid].pos = pos
                        self._elements[eid].quat = quat
                r, p, y = _quat_to_euler_deg(quat)
                pos_handle.value = (float(pos[0]), float(pos[1]), float(pos[2]))
                roll_handle.value = round(r, 1)
                pitch_handle.value = round(p, 1)
                yaw_handle.value = round(y, 1)

            gizmo.on_update(_gizmo_moved)

        @remove_btn.on_click
        def _(_) -> None:
            self._remove_element(eid)
            folder.remove()

        self._element_gui_folders[eid] = folder

    # ------------------------------------------------------------------
    # Removal and clear
    # ------------------------------------------------------------------

    def _remove_element(self, eid: str) -> None:
        """Remove element *eid* from the scene and all Viser handles.

        Args:
            eid: Element ID to remove.
        """
        with self._lock:
            self._elements.pop(eid, None)

        gizmo = self._element_gizmos.pop(eid, None)
        if gizmo is not None:
            gizmo.remove()

        for handle in self._element_visuals.pop(eid, []):
            handle.remove()

        self._element_gui_folders.pop(eid, None)

    def _undo_last(self) -> None:
        """Remove the most recently added element that has not already been removed.

        Skips over any undo-stack entries whose element was already manually
        deleted, so repeated undo steps walk backwards through the full add
        history.
        """
        while self._undo_stack:
            eid = self._undo_stack.pop()
            if eid in self._elements:
                folder = self._element_gui_folders.get(eid)
                self._remove_element(eid)
                if folder is not None:
                    folder.remove()
                return

    def _clear_all(self) -> None:
        """Remove every element from the scene."""
        eids = list(self._elements.keys())
        for eid in eids:
            folder = self._element_gui_folders.pop(eid, None)
            if folder is not None:
                folder.remove()
            self._remove_element(eid)
        self._undo_stack.clear()

    # ------------------------------------------------------------------
    # MJCF spec construction
    # ------------------------------------------------------------------

    def build_spec(self) -> mujoco.MjSpec:
        """Sync gizmo positions and compile all elements into a :class:`mujoco.MjSpec`.

        Returns:
            A fully composed MjSpec ready for ``.compile()`` or ``.to_xml()``.
        """
        self._sync_positions()

        spec = mujoco.MjSpec()
        spec.modelname = "web_composed_scene"

        preset = SCENE_PRESETS[self._active_scene]
        self._add_scene_to_spec(spec, preset)

        with self._lock:
            elements = list(self._elements.values())

        for elem in elements:
            if elem.kind == "fragment":
                self._attach_fragment_to_spec(spec, elem)
            elif elem.kind == "primitive":
                self._attach_primitive_to_spec(spec, elem)
            elif elem.kind == "mesh":
                self._attach_mesh_to_spec(spec, elem)
            elif elem.kind == "mimic":
                self._attach_mimic_to_spec(spec, elem)

        return spec

    def _add_scene_to_spec(self, spec: mujoco.MjSpec, preset: ScenePreset) -> None:
        """Add floor, pedestal, and lighting to *spec* from *preset*.

        Args:
            spec: Target MjSpec.
            preset: Scene preset to materialise.
        """
        light = spec.worldbody.add_light()
        light.pos = [0.0, -4.0, 4.0]
        light.dir = [0.0, 1.0, -1.0]
        light.diffuse = [0.8, 0.8, 0.8]
        light.specular = [0.2, 0.2, 0.2]

        if preset.has_floor:
            floor = spec.worldbody.add_geom(name="floor")
            floor.type = mujoco.mjtGeom.mjGEOM_PLANE
            floor.size = [5.0, 5.0, 0.1]
            floor.pos = [0.0, 0.0, float(preset.floor_z)]
            floor.rgba = [0.85, 0.85, 0.85, 1.0]
            floor.conaffinity = 1
            floor.contype = 1

        if preset.has_pedestal:
            ped = spec.worldbody.add_geom(name="pedestal")
            ped.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            ped.size = [1.05, 0.205, 0.0]
            ped.pos = [0.0, 0.0, float(preset.floor_z) + 0.205]
            ped.rgba = [0.5, 0.5, 0.5, 1.0]
            ped.contype = 1
            ped.conaffinity = 1

    def _attach_fragment_to_spec(self, spec: mujoco.MjSpec, elem: SceneElement) -> None:
        """Attach an MSK fragment XML into *spec* using the element's gizmo pose.

        Each fragment is namespaced with its ``eid`` as a prefix, so multiple
        instances of the same fragment coexist without name collisions.

        Args:
            spec: Target MjSpec.
            elem: Fragment element with ``fragment_name`` and pose.
        """
        if elem.fragment_name is None:
            return
        path = (
            get_simhive_asset_root("myo_sim") / FRAGMENT_CATALOGUE[elem.fragment_name]
        )
        if not path.exists():
            return

        frag_spec = mujoco.MjSpec.from_file(str(path))
        frame = spec.worldbody.add_frame()
        frame.pos = elem.pos.tolist()
        frame.quat = elem.quat.tolist()
        spec.attach(frag_spec, prefix=f"{elem.eid}_", suffix="", frame=frame)

    def _attach_primitive_to_spec(
        self, spec: mujoco.MjSpec, elem: SceneElement
    ) -> None:
        """Add a free-body primitive geom to *spec*.

        Args:
            spec: Target MjSpec.
            elem: Primitive element.
        """
        if elem.primitive_type is None or elem.geom_size is None:
            return

        body = spec.worldbody.add_body(name=elem.eid)
        body.pos = elem.pos.tolist()
        body.quat = elem.quat.tolist()
        body.add_freejoint(name=f"{elem.eid}_free")

        geom = body.add_geom(name=f"{elem.eid}_geom")
        geom.type = _GEOM_TYPE_MAP.get(elem.primitive_type, mujoco.mjtGeom.mjGEOM_BOX)
        geom.size = elem.geom_size.tolist()
        if elem.rgba is not None:
            geom.rgba = elem.rgba.tolist()
        if elem.mass is not None:
            geom.mass = elem.mass

    def _attach_mesh_to_spec(self, spec: mujoco.MjSpec, elem: SceneElement) -> None:
        """Add a mesh free-body to *spec* with optional texture.

        Args:
            spec: Target MjSpec.
            elem: Mesh element.
        """
        if elem.mesh_path is None or not elem.mesh_path.exists():
            return

        scale = (
            elem.geom_size.tolist() if elem.geom_size is not None else [1.0, 1.0, 1.0]
        )

        mesh_asset = spec.add_mesh()
        mesh_asset.name = f"{elem.eid}_mesh"
        mesh_asset.file = str(elem.mesh_path)
        mesh_asset.scale = scale

        mat = spec.add_material()
        mat.name = f"{elem.eid}_mat"
        rgba = elem.rgba if elem.rgba is not None else np.array([0.8, 0.8, 0.8, 1.0])
        mat.rgba = rgba.tolist()

        if elem.texture_path is not None and elem.texture_path.exists():
            tex = spec.add_texture()
            tex.name = f"{elem.eid}_tex"
            tex.type = mujoco.mjtTexture.mjTEXTURE_2D
            tex.file = str(elem.texture_path)
            mat.textures[0] = tex.name

        body = spec.worldbody.add_body(name=elem.eid)
        body.pos = elem.pos.tolist()
        body.quat = elem.quat.tolist()
        body.add_freejoint(name=f"{elem.eid}_free")

        geom = body.add_geom(name=f"{elem.eid}_geom")
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = f"{elem.eid}_mesh"
        geom.material = f"{elem.eid}_mat"

    def _attach_mimic_to_spec(self, spec: mujoco.MjSpec, elem: SceneElement) -> None:
        """Attach a MuscleMimic model into *spec* using the element's gizmo pose.

        The model's scene include is stripped before attaching so only the
        body/muscle content is merged.  Each instance is namespaced with
        ``eid`` as a prefix to avoid collisions.

        Args:
            spec: Target MjSpec.
            elem: Mimic element with ``mimic_xml_path`` and pose.
        """
        if elem.mimic_xml_path is None or not elem.mimic_xml_path.exists():
            return

        try:
            mimic_spec = _load_musclemimic_spec(elem.mimic_xml_path)
        except Exception as exc:
            logging.warning(
                "Failed to load MuscleMimic spec %s: %s", elem.mimic_xml_path, exc
            )
            return

        frame = spec.worldbody.add_frame()
        frame.pos = elem.pos.tolist()
        frame.quat = elem.quat.tolist()
        spec.attach(mimic_spec, prefix=f"{elem.eid}_", suffix="", frame=frame)

    # ------------------------------------------------------------------
    # Position synchronisation
    # ------------------------------------------------------------------

    def _sync_positions(self) -> None:
        """Read each gizmo's current transform and write it back to the element record."""
        with self._lock:
            eids = list(self._elements.keys())

        for eid in eids:
            gizmo = self._element_gizmos.get(eid)
            if gizmo is None:
                continue
            with self._lock:
                if eid in self._elements:
                    self._elements[eid].pos = np.array(gizmo.position)
                    self._elements[eid].quat = np.array(gizmo.wxyz)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _do_export(self) -> None:
        """Internal callback: read export path from GUI and call ``export_mjcf``."""
        path_str = (
            self._export_path_handle.value if self._export_path_handle else "scene.xml"
        )
        self.export_mjcf(Path(path_str))

    def export_mjcf(self, path: str | Path) -> None:
        """Build the composed scene and write it to *path* as MJCF XML.

        Args:
            path: Destination file path (parent directories are created
                  automatically; file is overwritten if it exists).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        spec = self.build_spec()
        xml = spec.to_xml()
        path.write_text(xml, encoding="utf-8")
        print(f"[WebComposer] Exported: {path}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the Viser server and block until Ctrl-C.

        Open ``http://localhost:<port>`` in a browser to access the UI.
        """
        port = getattr(self._server, "_port", 8084)
        print(f"\n{'='*52}")
        print("  MyoSuite Web Composer")
        print(f"  URL: http://localhost:{port}")
        print(f"{'='*52}\n")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[WebComposer] Shutting down.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MyoSuite Web Composer")
    parser.add_argument(
        "--port", type=int, default=8084, help="Viser server port (default: 8084)"
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Default directory for MJCF exports",
    )
    args = parser.parse_args()

    composer = WebComposer(port=args.port, export_dir=args.export_dir)
    composer.run()
