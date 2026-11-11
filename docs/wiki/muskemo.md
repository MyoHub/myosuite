# MuSkeMo interoperability

`myosuite.integrations.muskemo.export_model` produces a data-only anatomy JSON
for an optional Blender/MuSkeMo importer. MuJoCo resolves XML includes, defaults,
mesh transforms, inertias and actuator classes before export. No Blender package
is imported by MyoSuite or added to its runtime dependencies.

```python
import myo_sim
from myosuite.integrations.muskemo import export_model, joint_sweep

model = myo_sim.load_spec("myoelbow").compile()
poses = joint_sweep(model, "r_elbow_flex", samples=21)
export_model(model, "elbow.json", qpos=poses)
```

The versioned document contains body-local COM/inertia, resolved visual meshes,
all joint types, site attachments, ordered tendon elements (including wraps,
sidesites and pulleys), muscle parameters and equality definitions. Poses are
evaluated with the shared joint-coupling projection and MuJoCo forward dynamics.
Each frame carries body/geometry transforms, sites, native actuator lengths and
native moment arms (`dL/dq`, including gear, over all MuJoCo velocity DOFs).
For dependent coordinates these are partial derivatives; the derivative along
an independent-coordinate sweep also includes the coupling chain rule.

Metres, radians and kilograms are used throughout, with wxyz quaternions. Mesh
vertices/faces are embedded, so Blender needs neither MuJoCo nor access to the
source model's mesh paths. Flexes, skins and heightfields fail explicitly.

The selected-edit/report extension also retains raw source names, actuator gear
and physical tendon lengths separately from geared actuator lengths. Generated
labels for unnamed objects are display labels, not editable source names.

This interchange is for anatomy authoring and geometric validation. It is not a
complete simulation serialization or a replacement for the original `MjSpec`:
preserve the original model when applying selected edits. The fingerprint binds
an export to that exact compiled model and MuJoCo version.

MuSkeMo is installed separately from
[its upstream repository](https://github.com/PashavanBijlert/MuSkeMo). Its code
and bundled geometry-node assets are not redistributed here.
