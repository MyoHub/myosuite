# Credits

## `myofullbody.skn`

- **What:** MuJoCo binary skin (`.skn`) of the human body surface: 14,517 vertices with UVs,
  26,756 triangles, and 58 bones.
- **Source:** adapted from the original [MakeHuman](http://www.makehumancommunity.org/) /
  [MPFB](https://github.com/makehumancommunity/mpfb2) mesh, rig and weights, released under
  [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
- **Adaptation:** the bones are bound to the bodies of the MyoSuite full-body model
  ([myo_sim](https://github.com/MyoHub/myo_sim), `myoMimicFullbody-v0`).
- **Contributed by:** Vittorio Caggiano, for the MyoSuite authors.
- **License:** the CC0 source places no restrictions on reuse; the adapted file is distributed
  with MyoSuite under the repository's Apache-2.0 license ([LICENSE](../../../LICENSE)).
- **Used by:** `myosuite.viz.skin` (`load_skn("fullbody")`) and `scripts/render_blender.py --skin fullbody`.

## Anatomical muscle meshes (`--muscle-mesh atlas` / `atlas-hd`)

- **What:** muscle surfaces of the whole body (`fullbody_muscles.glb`, ~334k vertices, and a
  decimated `fullbody_muscles_light.glb`), posed on the rest pose of `myoMimicFullbody-v0`.
- **Where:** not in this repository or the wheel; downloaded on first use from the Hugging Face
  dataset [`myohub/myosuite-assets`](https://huggingface.co/datasets/myohub/myosuite-assets)
  (`muscles/`).
- **Source:** [BodyParts3D](https://lifesciencedb.jp/bp3d/), © The Database Center for Life Science
  (DBCLS), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Mitsuhashi N,
  et al. *BodyParts3D: 3D structure database for anatomical concepts.* Nucleic Acids Res.
  2009;37:D782-5. [doi:10.1093/nar/gkn613](https://doi.org/10.1093/nar/gkn613). Obtained through
  [human-atlas](https://github.com/ashemag/human-atlas) (MIT licence).
- **Changes:** posed on the MyoSuite full-body skeleton, merged into two meshes
  (`Muscles_actuated`, `Muscles_no_actuator`), and decimated for the light copy.
- **Used by:** `myosuite.viz.blender_render` (`RenderConfig.muscle_mesh`, `--muscle-mesh`).

## Acknowledgements (ideas only, no code used)

- **MuSkeMo** — P. A. van Bijlert, *MuSkeMo: Open-source software to construct, analyze, and
  visualize human and animal musculoskeletal models and movements in Blender*, bioRxiv (preprint),
  2024, [doi:10.1101/2024.12.10.627828](https://doi.org/10.1101/2024.12.10.627828);
  <https://github.com/PashavanBijlert/MuSkeMo>. Its volume-accurate muscle visualiser in Blender
  inspired the look of `myosuite.viz.blender_render` (fusiform bellies with tendons, volume kept
  as a muscle shortens). MuSkeMo publishes no licence, so none of its code is used or adapted:
  `myosuite.viz.muscle_tubes` is an independent numpy implementation that builds the tubes and
  computes their volumes in closed form, without Blender Geometry Nodes.
- **Blender Stack Exchange** — the `calc_volume` geometry node by user bebop_artist,
  [answer 325516](https://blender.stackexchange.com/a/325516) to "How do you measure the volume of
  a mesh?" (CC BY-SA 4.0), which MuSkeMo uses to measure muscle mesh volumes. MyoSuite does not
  use it: its tube volumes are analytic.
