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
