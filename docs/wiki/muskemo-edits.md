# Selected anatomy edit API

Apply selected edits to a copied `MjSpec`, preserving the original model.
The [anatomy interchange](muskemo.md) binds authoring data to that exact source.

A schema-1 patch has exactly these fields; copy the fingerprint and MuJoCo
version from the anatomy export. Both edit lists are present, even when empty:

```json
{
  "schema": 1,
  "fingerprint": "<the source export's SHA-256>",
  "mujoco_version": "<the source export's version>",
  "sites": [{"name": "BRA-P2", "pos": [-0.002, -0.024, 0.001]}],
  "inertias": [{"name": "r_ulna_radius_hand", "mass": 1.6,
                "com": [0.0, -0.18, 0.0],
                "inertia": [[0.02, 0.001, 0.0], [0.001, 0.03, 0.0], [0.0, 0.0, 0.04]]}]
}
```

Site `pos` and inertial `com` use the named object's **body-local axes**, in metres.
Sites inside source frames are moved into a body-local identity frame while
preserving their compiled orientation; unsupported or ignored edits are rejected.
Sources whose compilation actually fuses static bodies are rejected for patches
and archives: native recompilation can lose their transforms. Disable
`fusestatic` before exporting such a source again. Merely enabling the option
without any body fusion remains supported.
Mass is kg. The 3×3 inertia is about the COM, expressed in the body's axes, in
kg m². It must be symmetric positive definite and its principal moments must
satisfy the triangle inequality. Names must identify existing named source
sites/bodies; the world inertia, duplicate edits and unknown fields are refused.
Changing joints, wraps, geometries, muscle parameters or compiler options is
outside this selected-edit schema. A changed source or MuJoCo version requires
a new export and patch. Mesh paths and all compiled model settings participate
in the source fingerprint; this is an exact-source check, not a cross-version
model-equivalence test.

```python
import myo_sim
from myosuite.integrations.muskemo_edits import apply_model_patch

source = myo_sim.load_spec("myoelbow")
edited = apply_model_patch(source, patch)  # validates a compiled copy
model = edited.compile()
```

The caller's spec is unchanged on success or failure. Compiler settings that
ignore or alter an explicit inertia cause rejection. Muscle calibration and
other bodies' settings are preserved: a site edit changes its tendon geometry,
but it does not silently recalibrate muscle fibre/slack lengths or checkpoint
contracts. Recheck lengths, moment arms and task/checkpoint compatibility.
For an explicitly homogeneous solid, install `myosuite[muskemo]` and use
`mesh_inertia(path, density)` to obtain `mass`, `com` and `inertia` fields via
trimesh. The input mesh must be watertight, consistently oriented, in body-local
metres, with positive density in kg/m³. Coordinates are not inferred or scaled.
The caller must rule out intersecting/overlapping solids and justify density;
the function does not infer tissues, repair holes or change the mesh.
