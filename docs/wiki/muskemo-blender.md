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
  --addon C:/path/to/MuSkeMo --output elbow.blend
```

To use the file picker, enable MuSkeMo and run `muskemo_blender.py` in Blender's
Text Editor. File > Import > MyoSuite anatomy accepts the same exported JSON.
Import namespaces prevent accidental replacement of existing work. A failed
import removes the objects/collections it created. Source names and IDs are
stored on objects, and compact source metadata is embedded in the blend file.

MuSkeMo wrapping is a preview using a different geometry algorithm. Sphere wraps
can differ substantially from MuJoCo. Original wrap definitions and sidesite IDs
remain in metadata for validation against the source model. Polynomial couplings
are preserved through baked poses, rather than live Blender joint constraints.
Pulley/fixed muscle paths, flexes, skins and heightfields fail explicitly.

Real Blender tests are opt-in:

```powershell
$env:MYOSUITE_BLENDER = 'C:/Program Files/Blender Foundation/Blender 5.3/blender.exe'
$env:MUSKEMO_ADDON = 'C:/path/to/MuSkeMo'
python -m pytest myosuite/tests/test_muskemo_blender.py
```

The tests check all elbow/arm/leg muscle counts, source object transforms,
wrapping references, baked elbow motion and duplicate-import protection.
