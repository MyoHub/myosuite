# Asset credits

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
