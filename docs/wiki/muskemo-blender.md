# Compiled MuJoCo importer for Blender and MuSkeMo

The importer uses separately installed MuSkeMo creation helpers. It imports all
muscle actuator classes from compiled MuJoCo data, resolved mesh transforms,
body-local inertias, site attachments, wrapping geometry and original tendon
order. Free/ball/hinge/slide joints retain their source definitions. Polynomial
equalities are evaluated by MuJoCo before export, with poses baked into Blender
body transforms. They are not replaced with constant translations.

Export a document using [muskemo.md](muskemo.md), then run:

```powershell
& 'C:/Program Files/Blender Foundation/Blender 5.3/blender.exe' `
  --background --factory-startup --python-exit-code 1 `
  --python myosuite/integrations/muskemo_blender.py -- elbow.json `
  --addon C:/path/to/MuSkeMo --output elbow.blend --report geometry.json
```

To use the file picker, enable MuSkeMo and run `muskemo_blender.py` in Blender's
Text Editor. File > Import > MyoSuite anatomy accepts the same exported JSON.
Import namespaces prevent accidental replacement of existing work. A failed
import removes the objects/collections it created. Source names and IDs are
stored on objects, and compact source metadata is embedded in the blend file.

The geometry report compares actual evaluated MuSkeMo lengths against MuJoCo at
every exported pose. With a scalar joint sweep it also compares effective moment
arms by finite differences. The source model remains the simulation authority:
MuSkeMo's wrapping uses a different algorithm, and some sphere wraps differ
substantially. For example, in the bundled elbow the BIClong sphere preview is
about 55 mm too long even though the other five muscles match within 1 micrometre
in the checked five-pose sweep. The importer exposes this discrepancy; do not
use Blender preview lengths to calibrate physiological parameters. Sidesite IDs
and all original wrap definitions remain available for source-model validation.
Pulley/fixed muscle paths, flexes, skins and heightfields fail explicitly.

To export a selected change, use `export_patch("edit.json")` from the importer
module. At frame 1 select edited site empties, edited muscle curves or bodies.
Body properties are `mass`, `COM_local` and `inertia_COM_local` (xx, yy, zz, xy,
xz, yz about the COM in body axes). Selected muscle attachments are converted
back to body-local sites. Conflicting shared-site edits are rejected. Apply the
versioned patch to the original `MjSpec`; no unrelated model properties change.

Real Blender tests are opt-in:

```powershell
$env:MYOSUITE_BLENDER = 'C:/Program Files/Blender Foundation/Blender 5.3/blender.exe'
$env:MUSKEMO_ADDON = 'C:/path/to/MuSkeMo'
python -m pytest myosuite/tests/test_muskemo_blender.py
```

The tests check all elbow/arm/leg muscle counts, source object transforms,
wrapping references, baked elbow motion and duplicate-import protection.
