# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""
Composable MJCF model builder using MuJoCo 3's MjSpec API.

.. warning::
    **Experimental API — subject to change.**
    ModelBuilder is under active development and does **not** yet replicate all
    features of the official MyoChallenge environment XML files.  For production
    training or evaluation against a challenge benchmark, load the environment
    directly via its registered Gymnasium ID (e.g. ``gym.make("myoChallengeBaodingP2-v1")``)
    or instantiate the environment class directly.

ModelBuilder is a lazy recipe — it records fragment attachments, prop bodies,
and MjSpec transforms without compiling until :meth:`build` is called.

Current limitations
-------------------
The following scene features are **not** supported natively and require an
:meth:`apply_transform` escape hatch (or direct XML loading):

* Tendons and spatial constraints (e.g. Baoding tracking tendons)
* Sites (tracking markers, sensor attachment points)
* Mocap bodies (static goal bins, opponent dummy, goal posts)
* Non-freejoint object kinematics (slide/hinge joints on manipulated objects)
* Multi-geom composite bodies (e.g. 12-capsule + 3-box die)
* Cube-map / grid-layout textures (e.g. soccer ball, die faces)
* Aerodynamic fluid coefficients (``fluidcoef`` on ping-pong ball)
* Mirrored body assembly (left arm in TableTennis)
* Heightfield terrain (RunTrack, ChaseTag)
* Contact filtering / exclusion pairs
* Keyframes (initial pose snapshots)
* Sensors (force, touch, accelerometer, …)

Alternatives
------------
* **Direct XML loading** — the recommended approach for full challenge parity::

      import mujoco
      model = mujoco.MjModel.from_xml_path("path/to/scene.xml")

* **Gymnasium entry points** — load a fully-configured challenge env::

      import gymnasium as gym
      env = gym.make("myoChallengeBaodingP2-v1")

* **apply_transform escape hatch** — extend ModelBuilder scenes with any
  MjSpec feature not covered above::

      def add_tendons(spec: mujoco.MjSpec) -> mujoco.MjSpec:
          # ... direct spec manipulation ...
          return spec

      model, spec = ModelBuilder().attach_fragment("hand").apply_transform(add_tendons).build()
"""

from __future__ import annotations


from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias
from collections.abc import Callable

import numpy as np
import mujoco

# Registry of named model recipes (populated by @model_recipe decorator)
_RECIPES: dict[str, Callable] = {}

_IDENTITY_QUAT: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])
_ZERO_POS: np.ndarray = np.zeros(3)

# Sequence-like inputs accepted by ModelBuilder placement helpers (tuple defaults).
_FloatVec3: TypeAlias = list[float] | tuple[float, float, float] | np.ndarray
_FloatVec4: TypeAlias = list[float] | tuple[float, float, float, float] | np.ndarray


@dataclass
class _FragmentSpec:
    """Internal record of a fragment attachment."""

    name: str
    parent: str = "worldbody"
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    spec: mujoco.MjSpec | None = field(default=None, compare=False)


@dataclass
class _FreeBodySpec:
    """Internal record of a free (floating) object body."""

    name: str
    geom_type: int  # mujoco.mjtGeom value
    geom_size: np.ndarray
    rgba: np.ndarray
    pos: np.ndarray
    quat: np.ndarray
    mass: float | None  # None → MuJoCo default (inferred from density)
    condim: int  # contact dimensionality (1, 3, 4, or 6)


@dataclass
class _MeshBodySpec:
    """Internal record of a free-floating body with a mesh geom and optional texture."""

    name: str
    mesh_file: Path
    pos: np.ndarray
    quat: np.ndarray
    scale: np.ndarray
    texture_file: Path | None
    rgba: np.ndarray
    material_shininess: float
    material_specular: float


def model_recipe(name: str) -> Callable:
    """Decorator to register a named model recipe.

    Args:
        name: Recipe identifier used in TaskConfig.model.

    Example:
        >>> @model_recipe("elbow_standard")
        ... def _elbow(b):
        ...     return b.attach_fragment("elbow")
    """

    def decorator(fn: Callable) -> Callable:
        _RECIPES[name] = fn
        return fn

    return decorator


def get_recipe(name: str) -> Callable:
    """Return a registered recipe function by name.

    Args:
        name: Recipe name registered via @model_recipe.

    Raises:
        KeyError: If the recipe name is not registered.
    """
    if name not in _RECIPES:
        raise KeyError(f"Unknown model recipe: {name!r}. Available: {list(_RECIPES)}")
    return _RECIPES[name]


def _resolve_fragment_path(name: str) -> Path:
    """Resolve a fragment name to an absolute XML path.

    Supports three naming conventions:

    1. **Plain name** (e.g. ``"elbow"``) — resolved via the legacy
       ``_FALLBACK_PATHS`` map or ``myo_sim.FragmentRegistry``.
    2. **Scoped name** (e.g. ``"kinematics/elbow"``, ``"parts/elbow"``) —
       used by :meth:`ModelBuilder.attach_kinematics` and
       :meth:`ModelBuilder.attach_part` to target the myo_sim Issue #75
       directory layout (``kinematics/<name>/body.xml``,
       ``parts/<name>/muscles.xml``).  Falls back to the plain fragment
       name when the new structure is not yet present in the submodule.
    3. **FragmentRegistry** (new myo_sim API) — tried first for all names.

    Args:
        name: Fragment name, optionally prefixed with ``"kinematics/"`` or
              ``"parts/"`` (Issue #75 scoped convention).

    Returns:
        Absolute path to the fragment XML file.

    Raises:
        FileNotFoundError: If the fragment cannot be resolved.
    """
    # Try new myo_sim package with FragmentRegistry
    try:
        import myo_sim  # type: ignore[import-untyped]

        if hasattr(myo_sim, "FragmentRegistry"):
            try:
                info = myo_sim.FragmentRegistry.get(name)
                return Path(info.path)
            except KeyError:
                # myo_sim is installed but this fragment is not registered there;
                # fall through to the simhive submodule fallback below.
                pass
    except ImportError:
        pass  # myo_sim not installed; use simhive submodule fallback

    here = Path(__file__).parent.parent  # myosuite/
    simhive = here / "simhive" / "myo_sim"

    # Issue #75 scoped convention: "kinematics/<name>" or "parts/<name>".
    # When the new directory structure lands in the submodule we resolve
    # directly; otherwise we strip the scope and fall back to the plain name.
    if "/" in name:
        scope, base = name.split("/", 1)
        if scope == "kinematics":
            candidate = simhive / "kinematics" / base / "body.xml"
            if candidate.exists():
                return candidate
        elif scope == "parts":
            candidate = simhive / "parts" / base / "muscles.xml"
            if candidate.exists():
                return candidate
        # Pre-Issue-#75 fallback: resolve the base name as a plain fragment.
        name = base

    # Fallback: look in the simhive submodule via the legacy path map.
    # PR #73 entries (arm_left, arms) will be live once the submodule is
    # updated to include myoarmL_body.xml / myoarms.xml.
    _FALLBACK_PATHS: dict[str, str] = {
        "elbow": "elbow/myoelbow_2dof6muscles.xml",
        "hand": "hand/myohand.xml",
        "finger": "finger/myofinger_v0.xml",
        "shoulder": "arm/myoarm.xml",
        "arm": "arm/myoarm.xml",
        # PR #73: left arm (all body names suffixed with _left) and
        # bimanual assembly (77 DOF, 126 muscles, right bodies get _r suffix).
        "arm_left": "arm/myoarmL_body.xml",
        "arms": "arm/myoarms.xml",
        "leg": "leg/myolegs.xml",
        "osl": "osl/myolegs_osl.xml",
        "body": "body/myobody.xml",
        "torso": "torso/myotorso.xml",
    }
    if name in _FALLBACK_PATHS:
        candidate = simhive / _FALLBACK_PATHS[name]
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot resolve fragment {name!r}. "
        "Install the myo-sim pip package (`pip install myo-sim`) "
        "or initialise the git submodule "
        "(`git submodule update --init myosuite/simhive/myo_sim`)."
    )


def _try_myo_sim_compose(name: str) -> mujoco.MjSpec | None:
    """Return a fragment-only MjSpec from myo_sim's compose pipeline, or None.

    Consults ``myo_sim._FRAGMENT_SPEC_BUILDERS`` so that names like ``"hand"``
    are transparently routed through the build-system compose path instead of
    falling back to a static bundled XML.  Returns None if myo_sim is not
    installed, is too old to have the registry, or the name is not a composed
    fragment.
    """
    try:
        import myo_sim  # type: ignore[import-untyped]

        builder = getattr(myo_sim, "_FRAGMENT_SPEC_BUILDERS", {}).get(name)
        return builder() if builder is not None else None
    except Exception:  # ImportError, compose failures, etc.
        return None


def _get_parent_body(spec: mujoco.MjSpec, parent: str) -> mujoco.MjsBody:
    """Return the MjsBody for *parent*, raising clearly if not found.

    Args:
        spec: The spec to search.
        parent: Body name, or ``"worldbody"`` for the root.

    Returns:
        The matching MjsBody.

    Raises:
        KeyError: If *parent* is not ``"worldbody"`` and cannot be found.
    """
    if parent == "worldbody":
        return spec.worldbody
    try:
        return spec.body(parent)
    except Exception as exc:
        raise KeyError(
            f"Parent body {parent!r} not found in spec. "
            "Attach the parent fragment first, or use parent='worldbody'."
        ) from exc


def _add_mesh_body_to_spec(spec: mujoco.MjSpec, mb: _MeshBodySpec) -> None:
    """Add a mesh body with optional texture/material to *spec* in-place.

    Args:
        spec: The target MjSpec to modify.
        mb: Mesh body descriptor.
    """
    # Register mesh asset
    mesh = spec.add_mesh()
    mesh.name = f"{mb.name}_mesh"
    mesh.file = str(mb.mesh_file)
    mesh.scale = mb.scale

    # Register material (with or without texture)
    mat = spec.add_material()
    mat.name = f"{mb.name}_mat"
    mat.rgba = mb.rgba
    mat.shininess = mb.material_shininess
    mat.specular = mb.material_specular

    if mb.texture_file is not None:
        tex = spec.add_texture()
        tex.name = f"{mb.name}_tex"
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.file = str(mb.texture_file)
        # Index 0 is the diffuse texture slot in MjSpec's MjStringVec
        mat.textures[0] = tex.name

    # Create the body with a freejoint and mesh geom
    body = spec.worldbody.add_body(name=mb.name)
    body.pos = mb.pos
    body.quat = mb.quat
    body.add_freejoint(name=f"{mb.name}_free")
    geom = body.add_geom(
        name=f"{mb.name}_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
    )
    geom.meshname = f"{mb.name}_mesh"
    geom.material = f"{mb.name}_mat"


class ModelBuilder:
    """Composable MJCF model builder.

    .. warning::
        **Experimental.**  See module docstring for the full list of unsupported
        scene features and recommended alternatives for production use.

    Records fragment attachments and transform functions lazily;
    compiles and caches the result on the first call to :meth:`build`.

    Positioning
    -----------
    Use :meth:`place_fragment` to attach a body-part at an explicit position
    and orientation in the scene.  Use :meth:`add_free_body` to add a
    free-floating prop (sphere, box, cylinder, …) at a given pose.
    Use :meth:`add_mesh_body` to add a free-floating body with a mesh and
    optional texture.  For anything else, :meth:`apply_transform` gives direct
    access to the compiled :class:`mujoco.MjSpec`.

    Examples:

        >>> # Attach elbow at origin (default)
        >>> model, spec = ModelBuilder().attach_fragment("elbow").build()

        >>> # Place hand 0.5 m above the origin, rotated 90° around Z
        >>> import numpy as np
        >>> model, spec = (
        ...     ModelBuilder()
        ...     .place_fragment("hand", pos=[0, 0, 0.5], quat=[0.707, 0, 0, 0.707])
        ...     .build()
        ... )

        >>> # Add a red sphere prop at (0.3, 0, 0.1)
        >>> model, spec = (
        ...     ModelBuilder()
        ...     .attach_fragment("hand")
        ...     .add_free_body("target_ball", pos=[0.3, 0, 0.1],
        ...                    geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
        ...                    geom_size=[0.02, 0, 0], rgba=[1, 0, 0, 1])
        ...     .build()
        ... )

        >>> # Add a mesh object with texture
        >>> model, spec = (
        ...     ModelBuilder()
        ...     .attach_fragment("hand")
        ...     .add_mesh_body("ball", mesh_file="/path/to/ball.obj",
        ...                    pos=[0.3, 0, 0.1], texture_file="/path/to/ball.png")
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        self._fragments: list[_FragmentSpec] = []
        self._free_bodies: list[_FreeBodySpec] = []
        self._mesh_bodies: list[_MeshBodySpec] = []
        self._transforms: list[Callable[[mujoco.MjSpec], mujoco.MjSpec]] = []
        self._timestep: float | None = None
        self._disable_contacts: bool = False
        self._xml_path: Path | None = None  # set by from_xml_file()

    @classmethod
    def from_xml_file(cls, path: str | Path) -> ModelBuilder:
        """Create a builder seeded from an existing MJCF XML file.

        Use this when the model cannot be expressed as a composition of
        registered fragments — e.g. when loading a complete monolithic XML
        and then applying surgical spec mutations via :meth:`apply_transform`.

        Chaining example::

            model, spec = (
                ModelBuilder.from_xml_file("myoarm.xml")
                .apply_transform(edit_fn_arm_reaching)
                .build()
            )

        Args:
            path: Absolute or relative path to the MJCF XML file.

        Returns:
            A new :class:`ModelBuilder` instance with *path* as its seed.
        """
        from myosuite.utils.simhive_path import resolve_model_xml_path

        builder = cls()
        builder._xml_path = resolve_model_xml_path(Path(path))
        return builder

    def attach_fragment(self, name: str, parent: str = "worldbody") -> ModelBuilder:
        """Append a body-part XML fragment at the parent body's origin.

        To control placement, use :meth:`place_fragment` instead.

        Args:
            name: Fragment identifier (resolved via myo_sim.FragmentRegistry
                  or simhive fallback).
            parent: Name of the parent body to attach to.

        Returns:
            Self, for method chaining.
        """
        self._fragments.append(_FragmentSpec(name=name, parent=parent))
        return self

    def attach_spec(
        self,
        spec: mujoco.MjSpec,
        name: str = "<inline>",
        parent: str = "worldbody",
    ) -> ModelBuilder:
        """Attach a pre-built MjSpec fragment directly, bypassing file resolution.

        Use this when the fragment is composed programmatically rather than
        loaded from a static XML file — for example, a hand pruned from the
        full arm via ``myo_sim.build.compose.load_right_hand_from_arm_spec()``.

        Args:
            spec: A fully-constructed ``mujoco.MjSpec`` to attach.
            name: Label used in error messages and cache keys (not a file path).
            parent: Name of the parent body to attach to.

        Returns:
            Self, for method chaining.

        Example:
            >>> from myo_sim.build.compose import load_right_hand_from_arm_spec
            >>> model, spec = ModelBuilder().attach_spec(
            ...     load_right_hand_from_arm_spec(), name="hand"
            ... ).build()
        """
        self._fragments.append(_FragmentSpec(name=name, parent=parent, spec=spec))
        return self

    def attach_kinematics(self, name: str, parent: str = "worldbody") -> ModelBuilder:
        """Attach the kinematics-only component of a body part (Issue #75 API).

        Resolves to ``kinematics/<name>/body.xml`` in the myo_sim Issue #75
        directory layout (bodies, joints, and global sites only — no muscles or
        tendons).  Falls back transparently to the full ``<name>`` fragment when
        the new structure is not yet present in the submodule.

        Args:
            name: Body-part name (e.g. ``"elbow"``, ``"arm"``).
            parent: Name of the parent body to attach to.

        Returns:
            Self, for method chaining.
        """
        self._fragments.append(_FragmentSpec(name=f"kinematics/{name}", parent=parent))
        return self

    def attach_part(self, name: str, parent: str = "worldbody") -> ModelBuilder:
        """Attach the actuation component of a body part (Issue #75 API).

        Resolves to ``parts/<name>/muscles.xml`` in the myo_sim Issue #75
        directory layout (muscles, tendons, wrapping surfaces, and local sites).
        Falls back transparently to the full ``<name>`` fragment when the new
        structure is not yet present in the submodule.

        Args:
            name: Body-part name (e.g. ``"elbow"``, ``"arm"``).
            parent: Name of the parent body to attach to.

        Returns:
            Self, for method chaining.
        """
        self._fragments.append(_FragmentSpec(name=f"parts/{name}", parent=parent))
        return self

    def place_fragment(
        self,
        name: str,
        pos: _FloatVec3 = (0.0, 0.0, 0.0),
        quat: _FloatVec4 = (1.0, 0.0, 0.0, 0.0),
        parent: str = "worldbody",
    ) -> ModelBuilder:
        """Attach a fragment at an explicit position and orientation.

        The position and quaternion are relative to the *parent* body frame.
        The quaternion is in MJCF convention: ``[w, x, y, z]``.

        Args:
            name: Fragment identifier (same as :meth:`attach_fragment`).
            pos: 3-vector ``[x, y, z]`` offset from the parent origin (metres).
            quat: 4-vector ``[w, x, y, z]`` orientation quaternion.
            parent: Name of the parent body to attach to.

        Returns:
            Self, for method chaining.

        Example:
            >>> import numpy as np
            >>> builder = (
            ...     ModelBuilder()
            ...     .place_fragment("elbow", pos=[0, 0, 0])
            ...     .place_fragment("hand", pos=[0, 0, -0.35], parent="elbow_distal")
            ... )
        """
        self._fragments.append(
            _FragmentSpec(
                name=name,
                parent=parent,
                pos=np.asarray(pos, dtype=float),
                quat=np.asarray(quat, dtype=float),
            )
        )
        return self

    def add_free_body(
        self,
        name: str,
        pos: _FloatVec3 = (0.0, 0.0, 0.0),
        quat: _FloatVec4 = (1.0, 0.0, 0.0, 0.0),
        geom_type: int = mujoco.mjtGeom.mjGEOM_SPHERE,
        geom_size: _FloatVec3 = (0.02, 0.0, 0.0),
        rgba: _FloatVec4 = (0.8, 0.8, 0.2, 1.0),
        mass: float | None = None,
        condim: int = 3,
    ) -> ModelBuilder:
        """Add a free-floating prop body (sphere, box, cylinder, …) to the scene.

        The body gets a freejoint so it is fully unconstrained at episode start.
        Set its initial position at runtime via ``data.qpos[joint_addr:joint_addr+7]``.

        Args:
            name: Unique body name (also used as geom name and freejoint name).
            pos: Initial world-frame position ``[x, y, z]`` (metres).
            quat: Initial orientation quaternion ``[w, x, y, z]``.
            geom_type: MuJoCo geom type constant (default: sphere).
                       Use ``mujoco.mjtGeom.mjGEOM_BOX``,
                       ``mujoco.mjtGeom.mjGEOM_CYLINDER``, etc.
            geom_size: Geom size parameters — meaning depends on *geom_type*:
                       sphere ``[radius, 0, 0]``;
                       box ``[half-x, half-y, half-z]``;
                       cylinder ``[radius, half-height, 0]``.
            rgba: Colour and opacity ``[r, g, b, a]``.
            mass: Body mass in kg.  ``None`` lets MuJoCo infer from density.
            condim: Contact dimensionality (1, 3, 4, or 6).  Use 4 for
                    objects that require torsional friction (e.g. rolling balls).

        Returns:
            Self, for method chaining.

        Example:
            >>> import mujoco
            >>> builder = (
            ...     ModelBuilder()
            ...     .attach_fragment("hand")
            ...     .add_free_body("red_ball",   pos=[0.3, 0.0, 0.1],
            ...                    geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
            ...                    geom_size=[0.02, 0, 0], rgba=[1, 0, 0, 1],
            ...                    mass=0.0027, condim=4)
            ...     .add_free_body("blue_box",   pos=[0.0, 0.2, 0.05],
            ...                    geom_type=mujoco.mjtGeom.mjGEOM_BOX,
            ...                    geom_size=[0.03, 0.03, 0.03], rgba=[0, 0, 1, 1])
            ... )
        """
        self._free_bodies.append(
            _FreeBodySpec(
                name=name,
                geom_type=geom_type,
                geom_size=np.asarray(geom_size, dtype=float),
                rgba=np.asarray(rgba, dtype=float),
                pos=np.asarray(pos, dtype=float),
                quat=np.asarray(quat, dtype=float),
                mass=mass,
                condim=condim,
            )
        )
        return self

    def add_mesh_body(
        self,
        name: str,
        mesh_file: str | Path,
        pos: _FloatVec3 = (0.0, 0.0, 0.0),
        quat: _FloatVec4 = (1.0, 0.0, 0.0, 0.0),
        scale: _FloatVec3 = (1.0, 1.0, 1.0),
        texture_file: str | Path | None = None,
        rgba: _FloatVec4 = (0.8, 0.8, 0.8, 1.0),
        material_shininess: float = 0.5,
        material_specular: float = 0.3,
    ) -> ModelBuilder:
        """Add a free-floating body with a mesh geom and optional texture.

        The mesh file is loaded from disk at build time.  If *texture_file* is
        provided, a texture + material are created and applied to the geom;
        otherwise *rgba* is used as a flat colour.

        Args:
            name: Unique body name (also used as a prefix for the mesh, texture,
                  material, and freejoint names).
            mesh_file: Path to the ``.obj`` (or other MuJoCo-supported) mesh file.
            pos: Initial world-frame position ``[x, y, z]`` (metres).
            quat: Initial orientation quaternion ``[w, x, y, z]``.
            scale: Mesh scale factors ``[sx, sy, sz]`` applied at load time.
            texture_file: Optional path to a texture image (PNG, JPG, …).
                          When ``None``, the geom is rendered with *rgba* only.
            rgba: Flat colour and opacity ``[r, g, b, a]``.  Used as the geom
                  colour when no texture is provided, or as the material base
                  colour when a texture is provided.
            material_shininess: Shininess of the generated material (0–1).
            material_specular: Specular intensity of the generated material (0–1).

        Returns:
            Self, for method chaining.

        Raises:
            FileNotFoundError: If *mesh_file* or *texture_file* does not exist.

        Example:
            >>> builder = (
            ...     ModelBuilder()
            ...     .attach_fragment("hand")
            ...     .add_mesh_body(
            ...         "ping_pong_ball",
            ...         mesh_file="/path/to/ball.obj",
            ...         pos=[0.3, 0.0, 0.1],
            ...         texture_file="/path/to/ball.png",
            ...     )
            ... )
        """
        mesh_path = Path(mesh_file)
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        tex_path: Path | None = None
        if texture_file is not None:
            tex_path = Path(texture_file)
            if not tex_path.exists():
                raise FileNotFoundError(f"Texture file not found: {tex_path}")

        self._mesh_bodies.append(
            _MeshBodySpec(
                name=name,
                mesh_file=mesh_path,
                pos=np.asarray(pos, dtype=float),
                quat=np.asarray(quat, dtype=float),
                scale=np.asarray(scale, dtype=float),
                texture_file=tex_path,
                rgba=np.asarray(rgba, dtype=float),
                material_shininess=float(material_shininess),
                material_specular=float(material_specular),
            )
        )
        return self

    def apply_transform(
        self, fn: Callable[[mujoco.MjSpec], mujoco.MjSpec]
    ) -> ModelBuilder:
        """Register an arbitrary MjSpec transform.

        Args:
            fn: Callable that receives and returns a MjSpec.

        Returns:
            Self, for method chaining.
        """
        self._transforms.append(fn)
        return self

    def set_timestep(self, dt: float) -> ModelBuilder:
        """Set the simulation timestep.

        Args:
            dt: Simulation timestep in seconds.

        Returns:
            Self, for method chaining.
        """
        self._timestep = dt
        return self

    def disable_cylinder_contacts(self) -> ModelBuilder:
        """Disable contacts on cylindrical geoms (reduces simulation stiffness).

        Returns:
            Self, for method chaining.
        """
        self._disable_contacts = True
        return self

    def apply_sarcopenia(self, force_scale: float = 0.5) -> ModelBuilder:
        """Scale muscle peak forces to simulate sarcopenia.

        Args:
            force_scale: Fraction of original peak force to retain (0–1).

        Returns:
            Self, for method chaining.
        """
        from myosuite.core.muscle_conditions import apply_sarcopenia_to_spec

        def _sarco(spec: mujoco.MjSpec) -> mujoco.MjSpec:
            return apply_sarcopenia_to_spec(spec, force_scale=force_scale)

        return self.apply_transform(_sarco)

    def build(self) -> tuple[mujoco.MjModel, mujoco.MjSpec]:
        """Compile the spec into a MjModel.

        Returns:
            Tuple of (MjModel, MjSpec).

        Raises:
            FileNotFoundError: If a fragment XML cannot be resolved.
            KeyError: If a named parent body does not exist in the spec.
        """
        if self._xml_path is not None:
            if not self._xml_path.exists():
                raise FileNotFoundError(f"XML not found: {self._xml_path}")
            spec = mujoco.MjSpec.from_file(str(self._xml_path))
        else:
            spec = mujoco.MjSpec()
        if self._timestep is not None:
            spec.timestep = self._timestep

        for frag in self._fragments:
            if frag.spec is not None:
                frag_spec = frag.spec
            else:
                composed = _try_myo_sim_compose(frag.name)
                if composed is not None:
                    frag_spec = composed
                else:
                    path = _resolve_fragment_path(frag.name)
                    frag_spec = mujoco.MjSpec.from_file(str(path))
            parent_body = _get_parent_body(spec, frag.parent)
            frame = parent_body.add_frame()
            frame.pos = frag.pos
            frame.quat = frag.quat
            spec.attach(frag_spec, prefix="", suffix="", frame=frame)

        for fb in self._free_bodies:
            body = spec.worldbody.add_body(name=fb.name)
            body.pos = fb.pos
            body.quat = fb.quat
            geom = body.add_geom(
                name=fb.name,
                type=fb.geom_type,
                size=fb.geom_size,
                rgba=fb.rgba,
            )
            geom.condim = fb.condim
            if fb.mass is not None:
                geom.mass = fb.mass
            body.add_freejoint(name=fb.name + "_free")

        for mb in self._mesh_bodies:
            _add_mesh_body_to_spec(spec, mb)

        for transform in self._transforms:
            spec = transform(spec)

        return spec.compile(), spec


def list_recipes() -> list[str]:
    """Return the names of all registered model recipes.

    Returns:
        Sorted list of recipe name strings.

    Example:
        >>> from myosuite.core.model_builder import list_recipes
        >>> "elbow_standard" in list_recipes()
        True
    """
    return sorted(_RECIPES)


def build_from_recipe(name: str, **kwargs: Any) -> tuple[mujoco.MjModel, mujoco.MjSpec]:
    """Build a model from a named recipe.

    Args:
        name: Recipe name registered via @model_recipe.
        **kwargs: Unused; reserved for future recipe parameterisation.

    Returns:
        Tuple of (MjModel, MjSpec).

    Example:
        >>> model, spec = build_from_recipe("elbow_standard")
    """
    recipe_fn = get_recipe(name)
    builder = ModelBuilder()
    builder = recipe_fn(builder)
    return builder.build()
