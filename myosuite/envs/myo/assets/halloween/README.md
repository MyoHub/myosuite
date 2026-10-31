# MyoHalloween assets

## Pumpkin mesh (Kenney Food Kit)

| | |
| --- | --- |
| File | `pumpkin.obj` / `pumpkin.mtl` / `Textures/colormap.png` / `pumpkin.glb` |
| Source | [Kenney Food Kit](https://kenney.nl/assets/food-kit) |
| License | **CC0** — see `LICENSE_Kenney_FoodKit_CC0.txt` |
| Credit | Optional: Kenney (www.kenney.nl) |

Collision in the MuJoCo scene uses a sphere; the Kenney mesh is visual-only.

## Settled showcase reset state

`terra_initial_state.json` contains qpos, qvel and muscle activations extracted
without modification from the public `motion_decoder.npz` at MyoSuite revision
`8842a86052d46be13b489a1e755bd265bcbc78ae` (NPZ SHA-256
`b3f9a27ed5a9b3dd64f0efb8671eb5954fe1c0ce1ed681e3e5ba339d2468b78c`).
Only the opt-in TERRA profile uses it. It is a small reset-state data file,
not a policy or a motion decoder, so environment creation needs no download.
It retains the source data's terms; the Kenney CC0 license covers pumpkin assets.
