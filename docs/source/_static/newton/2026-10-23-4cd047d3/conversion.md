# Newton compatibility probe

newton 1.8.0.dev0, warp-lang 1.18.0, mujoco 3.15.0, mujoco-warp 3.15.0, musclemimic-models 1.0.6, myosuite 3.0.0. `SolverMuJoCo(use_mujoco_cpu=True)` on the CPU; rtol 1e-06, atol 1e-08; 40 s.

Cells read MuJoCo reference -> Newton. `authored` is the MJCF MyoSuite loads; `normalized` is MuJoCo's explicit rewrite of it (see `scripts/newton_compat_probe.py`). *no meshes*: Newton failed to load the meshes and the model was imported without them.

| model | input | result | actuators | muscles | muscles lost | length ranges off | max force diff (N) | max rel. force diff | non-finite forces |
|---|---|---|---|---|---|---|---|---|---|
| myoElbowPose1D6MRandom-v0 | authored | differences | 6 -> 6 | 6 -> 6 | 0 | 0/6 (max 1.11e-08 m) | 0.000249 | 3.25e-06 | 0 |
| myoElbowPose1D6MRandom-v0 | normalized | differences | 6 -> 6 | 6 -> 6 | 0 | 0/6 (max 1.11e-08 m) | 0.000256 | 2.16e-06 | 0 |
| myoHandPoseRandom-v0 | authored | differences | 39 -> 39 | 39 -> 39 | 0 | 0/39 (max 1.49e-08 m) | 0.00529 | 8.25e-05 | 0 |
| myoHandPoseRandom-v0 | normalized | differences | 39 -> 39 | 39 -> 39 | 0 | 0/39 (max 1.49e-08 m) | 0.00529 | 8.25e-05 | 0 |
| myoArmReachRandom-v0 | authored | differences | 50 -> 50 | 50 -> 50 | 0 | 0/50 (max 1.49e-08 m) | 0.00935 | 6.23e-05 | 0 |
| myoArmReachRandom-v0 | normalized | differences | 50 -> 50 | 50 -> 50 | 0 | 0/50 (max 1.49e-08 m) | 0.00935 | 6.23e-05 | 0 |
| myoLegWalk-v0 | authored | differences | 80 -> 80 | 80 -> 80 | 0 | 0/80 (max 2.87e-08 m) | 0.00257 | 3.47e-06 | 0 |
| myoLegWalk-v0 | normalized | differences | 80 -> 80 | 80 -> 80 | 0 | 0/80 (max 2.87e-08 m) | 0.00244 | 3.29e-06 | 0 |
| musclemimic:myofullbody | authored | differences | 416 -> 416 | 416 -> 416 | 0 | 0/416 (max 2.87e-08 m) | 0.00464 | 8.79e-05 | 0 |
| musclemimic:myofullbody | normalized | differences | 416 -> 416 | 416 -> 416 | 0 | 0/416 (max 2.87e-08 m) | 0.00417 | 9.72e-05 | 0 |
| musclemimic:bimanual | authored | differences | 126 -> 126 | 126 -> 126 | 0 | 0/126 (max 1.49e-08 m) | 0.00212 | 4.56e-05 | 0 |
| musclemimic:bimanual | normalized | differences | 126 -> 126 | 126 -> 126 | 0 | 0/126 (max 1.49e-08 m) | 0.00179 | 6.62e-05 | 0 |

| model | input | nbody | njnt | nq | nv | mass (kg) | tendons | wraps (geom wraps) | equalities | sensors | integrator | qpos0 max diff | names kept (body, joint, site, tendon, actuator) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| myoElbowPose1D6MRandom-v0 | authored | 29 -> 29 | 1 -> 1 | 1 -> 1 | 1 -> 1 | 18.6 -> 18.6 | 6 -> 6 | 38 -> 38 (6 -> 6) | 0 -> 0 | 0 -> 0 | euler -> euler | joints differ | 1/29, 0/1, 0/40, 6/6, 6/6 |
| myoElbowPose1D6MRandom-v0 | normalized | 29 -> 29 | 1 -> 1 | 1 -> 1 | 1 -> 1 | 18.6 -> 18.6 | 6 -> 6 | 38 -> 38 (6 -> 6) | 0 -> 0 | 0 -> 0 | euler -> euler | joints differ | 1/29, 0/1, 0/40, 6/6, 6/6 |
| myoHandPoseRandom-v0 | authored | 38 -> 38 | 23 -> 23 | 23 -> 23 | 23 -> 23 | 4.63 -> 4.63 | 39 -> 39 | 315 -> 315 (63 -> 63) | 0 -> 0 | 0 -> 0 | euler -> euler | joints differ | 1/38, 0/23, 0/496, 39/39, 39/39 |
| myoHandPoseRandom-v0 | normalized | 38 -> 38 | 23 -> 23 | 23 -> 23 | 23 -> 23 | 4.63 -> 4.63 | 39 -> 39 | 315 -> 315 (63 -> 63) | 0 -> 0 | 0 -> 0 | euler -> euler | joints differ | 1/38, 0/23, 0/496, 39/39, 39/39 |
| myoArmReachRandom-v0 | authored | 53 -> 53 | 18 -> 18 | 18 -> 18 | 18 -> 18 | 41.8 -> 41.8 | 54 -> 54 | 397 -> 397 (90 -> 90) | joint 11 -> joint 11 | 0 -> 0 | euler -> euler | joints differ | 1/53, 0/18, 0/1171, 54/54, 50/50 |
| myoArmReachRandom-v0 | normalized | 53 -> 53 | 18 -> 18 | 18 -> 18 | 18 -> 18 | 41.8 -> 41.8 | 54 -> 54 | 397 -> 397 (90 -> 90) | joint 11 -> joint 11 | 0 -> 0 | euler -> euler | joints differ | 1/53, 0/18, 0/1171, 54/54, 50/50 |
| myoLegWalk-v0 | authored | 16 -> 16 | 29 -> 29 | 35 -> 35 | 34 -> 34 | 75.2 -> 75.2 | 80 -> 80 | 324 -> 324 (36 -> 36) | joint 14 -> joint 14 | 4 -> 4 | euler -> euler | joints differ | 1/16, 0/29, 0/383, 80/80, 80/80 |
| myoLegWalk-v0 | normalized | 16 -> 16 | 29 -> 29 | 35 -> 35 | 34 -> 34 | 75.2 -> 75.2 | 80 -> 80 | 324 -> 324 (36 -> 36) | joint 14 -> joint 14 | 4 -> 4 | euler -> euler | joints differ | 1/16, 0/29, 0/383, 80/80, 80/80 |
| musclemimic:myofullbody | authored | 102 -> 102 | 123 -> 123 | 129 -> 129 | 128 -> 128 | 84.3 -> 84.3 | 424 -> 424 | 1872 -> 1872 (270 -> 270) | joint 51 -> joint 51 | 110 -> 110 | euler -> euler | joints differ | 1/102, 0/123, 0/2021, 424/424, 416/416 |
| musclemimic:myofullbody | normalized | 102 -> 102 | 123 -> 123 | 129 -> 129 | 128 -> 128 | 84.3 -> 84.3 | 424 -> 424 | 1872 -> 1872 (270 -> 270) | joint 51 -> joint 51 | 110 -> 110 | euler -> euler | joints differ | 1/102, 0/123, 0/2021, 424/424, 416/416 |
| musclemimic:bimanual | authored | 77 -> 77 | 76 -> 76 | 76 -> 76 | 76 -> 76 | 16.5 -> 16.5 | 134 -> 134 | 928 -> 928 (202 -> 202) | joint 22 -> joint 22 | 106 -> 106 | euler -> euler | joints differ | 1/77, 0/76, 0/1016, 134/134, 126/126 |
| musclemimic:bimanual | normalized | 77 -> 77 | 76 -> 76 | 76 -> 76 | 76 -> 76 | 16.5 -> 16.5 | 134 -> 134 | 928 -> 928 (202 -> 202) | joint 22 -> joint 22 | 106 -> 106 | euler -> euler | joints differ | 1/77, 0/76, 0/1016, 134/134, 126/126 |
