# Anatomy CLI and portable archives

The CLI combines anatomy export, the optional separately installed
[Blender/MuSkeMo importer](muskemo-blender.md), and the
[source-bound edit API](muskemo-edits.md). MuJoCo remains the simulation source
of truth: a Blender preview can differ from native wrapping. Inspect `--report`
and validate edited models in MuJoCo before using them for experiments.

```bash
python scripts/muskemo.py export --model myoelbow --joint-sweep r_elbow_flex --output elbow.json
python scripts/muskemo.py blender elbow.json --addon /path/to/MuSkeMo --blender /path/to/blender --output elbow.blend --report geometry.json
python scripts/muskemo.py apply --model myoelbow --patch edits.json --output edited-elbow.zip
```

`export` and `apply` also accept `--xml source.xml` (or a native model zip) instead
of `--model`. Existing outputs are protected unless `--overwrite` is given.
The CLI fills MuJoCo's virtual filesystem with source file assets, then uses
`MjSpec.to_zip` and recompiles with `MjSpec.from_zip` before publishing the output.
It rejects ambiguous asset basenames or any geometry change during bundling.
Asset resolution follows MuJoCo's `strippath` setting for disk and virtual files.
Native XML serialization can introduce rounding; the elbow archive tests retain
geometry within 1e-6 and tendon length/moment curves within 1e-12.

Native `to_zip` alone includes only assets already in the spec's virtual
filesystem; the CLI also bundles its external files. Sources whose compilation
actually fuses static bodies are refused transactionally because native
recompilation can lose their transforms. Disable `fusestatic` before exporting
such a source again; enabling it without body fusion remains supported.
