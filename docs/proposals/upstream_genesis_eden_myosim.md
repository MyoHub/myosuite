# Upstream Engineering Proposals: Running MyoSuite on `genesis-world`, `Eden`, `myo_sim`, and `mjswan`

This document specifies the exact engineering blockers and PR-ready upstream proposals across:

1. **[`Genesis-Embodied-AI/Genesis`](https://github.com/Genesis-Embodied-AI/Genesis)** (`genesis-world`, audited at `v1.4.6`)
2. **[`embodied-ai-nexus/Eden`](https://github.com/embodied-ai-nexus/Eden)** (`Apache-2.0`)
3. **[`MyoHub/myo_sim`](https://github.com/MyoHub/myo_sim)** (`v0.2.5`)
4. **[`ttktjmt/mjswan`](https://github.com/ttktjmt/mjswan)** (Browser WebAssembly ONNX viewer)

---

## 1. What Does It Take for MyoSuite to Use `genesis-world`?

Why would MyoSuite want `genesis-world` alongside its existing CPU-MuJoCo, MJX, and `mjlab` (`mujoco-warp`) backends?
- **Multi-physics coupling:** Genesis couples rigid bodies with MPM soft tissue / deformable objects, PBD cloth, SPH/FEM fluids, and tactile/ray-traced sensors in a single Quadrants/Taichi GPU step — enabling musculoskeletal hands (`myoHand` / `myoArm`) manipulating deformable dough, liquids, soft prosthetics, or compliant orthoses.
- **Ecosystem interoperability:** Direct access to `Eden`'s scene generation, motion retargeting, and RL/VLA pipelines.

### Exact Audit of `genesis-world==1.4.6` (`genesis/utils/mjcf.py` & `RigidSolver`)

What Genesis **already supports** out of the box:
- Articulated MJCF kinematic trees (`mjJNT_FREE`, `mjJNT_BALL`, `mjJNT_HINGE`, `mjJNT_SLIDE`), inertias, collision geoms, and elliptic/pyramidal friction cones.
- Quartic polynomial joint equality constraints (`mjEQ_JOINT` — required by `myoArm`'s 11 shoulder/scapula couplings and `myoLeg`'s patella/knee couplings), plus `mjEQ_WELD` and `mjEQ_CONNECT`.
- Batched external generalized force injection via `RigidEntity.control_dofs_force(force, dofs_idx_local=...)`.

The **five exact blockers** preventing `gs.morphs.MJCF(file=...)` from loading a native MyoSuite model today:

| # | Blocker | Exact Upstream Location (`v1.4.6`) | Failure Mode on `myoHand` / `myoArm` / `myoFullBody` |
| :- | :--- | :--- | :--- |
| **B1** | **MuJoCo >= 3.17 `eq_data` width mismatch (12 vs 11)** | `genesis/utils/mjcf.py:720` vs `genesis/utils/urdf.py` & `RigidSolver._init_equality_fields` | MuJoCo 3.17 widened `mjModel.eq_data` from `11` to `12` columns (`mjNEQDATA=12`), while Genesis's URDF/equality buffers hardcode `11` columns, raising a broadcast `ValueError` in hybrid MJCF+URDF scenes. |
| **B2** | **1-Actuator-per-Joint assumption** | `genesis/utils/mjcf.py:421–478` (`(i_a,) = np.nonzero(actuator_mask_j)[0]`) | Genesis stores actuator gains directly on `RigidJoint`. Any joint crossed by >1 muscle raises `ValueError: too many values to unpack`; polyarticular muscles spanning 2–6 joints (`biceps`, `gastroc`, `EDC`) cannot be represented at all. |
| **B3** | **Spatial tendons rejected** | `genesis/utils/mjcf.py:435` (`(mj.wrap_type == mjWRAP_JOINT).all()`) | Every MyoSuite muscle routes through a spatial `<tendon>` composed of `<site>` waypoints and `<geom>` sphere/cylinder wraps (`mjWRAP_SITE`, `mjWRAP_SPHERE`, `mjWRAP_CYLINDER`, `sidesite`). Genesis drops all non-fixed tendons with a warning and discards their actuators. |
| **B4** | **No `RigidSite` forward-kinematics table** | `genesis/utils/mjcf.py:778` | Sites are not parsed into `RigidSolver` state (`site_xpos`, `site_xmat`), blocking both spatial tendon attachment points and MyoSuite site-tracking observations (`S_grasp`, `THtip`..`LFtip`, mimic body sites). |
| **B5** | **No Hill-type muscle dynamics or force curves** | `genesis/utils/mjcf.py:448–455` & `RigidSolver` actuation kernel | Only `mjDYN_NONE` (`PD`/`motor`), `mjGAIN_FIXED`/`AFFINE`, and `mjBIAS_NONE`/`AFFINE` are implemented. `mjDYN_MUSCLE` (Millard 1st-order activation ODE with `act` state), `mjGAIN_MUSCLE` (bump FLV curve), and `mjBIAS_MUSCLE` (passive quadratic/linear curve) raise/warn and fall back to zero gain. |

---

### Two Concrete Pathways to Run MyoSuite on `genesis-world`

#### Pathway 1 — Immediate Zero-Fork User-Space Bridge (**Implemented in This Repo**)
Available **today** against unmodified `genesis-world>=1.4.6`:
1. [`myosuite/physics/hill_muscle.py`](../../myosuite/physics/hill_muscle.py):
   - `strip_muscles_for_joint_torque_sim(spec)` clones any `MjSpec`, deletes `<tendon>` and muscle `<actuator>` elements, preserves all bodies, geoms, sites, contacts, and `mjEQ_JOINT` quartic couplings, and adds 1-to-1 unit-gear `mjTRN_JOINT` torque motors on independent DOFs so `genesis/utils/mjcf.py` loads the skeleton cleanly without triggering B2/B3/B5.
   - Implements exact float64/float32 parity (`~1e-15` vs `mujoco.mj_forward`) for `muscle_activation_dot`, `step_muscle_activation`, `muscle_gain`, `muscle_bias`, and `HillMuscleModel.generalized_forces`.
2. [`training/eden_rl/eden_rl/genesis_bridge.py`](../../training/eden_rl/eden_rl/genesis_bridge.py):
   - `GenesisMuscleBridge` steps batched `(B, nu)` muscle activations in PyTorch on GPU, evaluates `(length, velocity, moment_arm)` via either a pure-PyTorch analytical tendon-wrap callback (such as [`training/supermimic/supermimic/muscle_geometry.py`](../../training/supermimic/supermimic/muscle_geometry.py) + `torch.func.jacrev`) or MuJoCo FK, computes `tau = J_muscle^T @ f_muscle`, and injects `entity.control_dofs_force(tau)` every physics substep.

#### Pathway 2 — Native GPU Musculoskeletal Support in `Genesis-Embodied-AI/Genesis` (4 Stacked Upstream PRs)

To eliminate Python-side geometry callbacks and achieve full `gs.Scene.step()` fusion inside Quadrants/Taichi GPU kernels, propose the following **4 self-contained PRs** to `Genesis-Embodied-AI/Genesis`:

##### Genesis PR 1: Normalize `eq_data` Width Across MuJoCo 3.x (`<=3.16` vs `>=3.17`)
- **Files:** `genesis/utils/mjcf.py`, `genesis/utils/urdf.py`, `genesis/engine/solvers/rigid/rigid_solver_decomp.py`
- **Change:** Define `EQ_DATA_WIDTH = int(mujoco.mjNEQDATA)` (or canonical width `12`, right-padding 11-column URDF/programmatic `eq_data` arrays with `0.0` on slice `[:len(row)]`).
- **Impact:** Unblocks MuJoCo `>=3.17` coexistence across all MJCF + URDF scenes (~25 LOC).

##### Genesis PR 2: First-Class `RigidSite` Forward Kinematics Table
- **Files:** `genesis/utils/mjcf.py` (`parse_sites`), `genesis/engine/entities/rigid_entity/rigid_site.py`, `genesis/engine/solvers/rigid/rigid_solver_decomp.py`
- **Change:**
  - Parse `mj.nsite`, `mj.site_bodyid`, `mj.site_pos`, `mj.site_quat`, `mj.site_type`, `mj.site_size` into `sites_info`.
  - Update `sites_state.pos[i_s, i_b]` and `sites_state.quat[i_s, i_b]` at the end of `func_forward_kinematics`:
    ```python
    x_site = links_state.pos[body_id, i_b] + ti_transform_by_quat(sites_info.pos[i_s], links_state.quat[body_id, i_b])
    ```
  - Expose `entity.get_sites_pos()` / `entity.get_site(name)`.
- **Impact:** Unblocks MyoSuite fingertip/grasp/mimic observations AND provides endpoints for PR 3 (~140 LOC).

##### Genesis PR 3: Spatial Tendon Wrapping Kernel (`func_tendon`)
- **Files:** `genesis/utils/mjcf.py` (`parse_tendons`), `genesis/engine/solvers/rigid/tendon_decomp.py`
- **Change:**
  - Lift the `(mj.wrap_type == mjWRAP_JOINT).all()` gate in `mjcf.py:435`.
  - Port the closed-form 2-D tangent + cylinder-helix + inside-`sidesite` Newton wrap algorithm (matching `mujoco_warp/_src/util_misc.py:wrap` including [`google-deepmind/mujoco_warp#1713`](https://github.com/google-deepmind/mujoco_warp/pull/1713) inside-wrap parity, and mirrored in PyTorch in [`training/supermimic/supermimic/muscle_geometry.py`](../../training/supermimic/supermimic/muscle_geometry.py)) to a `@qd.kernel` computing:
    - `ten_length[i_t, i_b]`
    - `ten_J[i_t, dof, i_b]` (accumulating point-velocity Jacobians at segment endpoints; tangent points move rigidly with the wrap geom's parent link)
    - `ten_velocity[i_t, i_b] = sum_d ten_J[i_t, d, i_b] * dofs_state.vel[d, i_b]`
- **Impact:** Enables cable-driven dexterous hands, continuum robots, and all 290+ spatial muscle-tendon paths in `myoFullBody` (~350 LOC).

##### Genesis PR 4: Decoupled `RigidActuator` Table + Hill Muscle Dynamics (`mjDYN_MUSCLE` / `mjGAIN_MUSCLE` / `mjBIAS_MUSCLE`)
- **Files:** `genesis/utils/mjcf.py` (`parse_actuators`), `genesis/engine/solvers/rigid/rigid_solver_decomp.py`
- **Change:**
  - Decouple actuators `(nu,)` from DOFs `(nv,)` instead of folding `actuator_gainprm` into `RigidJoint` (`mjcf.py:421–478`). Keep existing per-DOF PD fast path for `mjTRN_JOINT` motors, and add a general actuator table for `mjTRN_TENDON` and multi-actuator joints.
  - Add `actuators_state.act[i_a, i_b]` buffer integrated with Millard's `mju_muscleDynamics` (`tau_act * (0.5 + 1.5*act)`, `tau_deact / (0.5 + 1.5*act)`, quintic `tausmooth` transition).
  - Port `mju_muscleGain` (`fl = mj_bump(L, lmin, 1.0, lmax)`, `fv` quadratic/hyperbolic segments) and `mju_muscleBias` (`b = 0.5 * (1.0 + lmax)`, scaled by `fpmax`) — using [`myosuite/physics/hill_muscle.py`](../../myosuite/physics/hill_muscle.py) as the float64 parity reference.
  - Project `dofs_state.qf_actuator[d, i_b] += sum_a actuator_moment[a, d, i_b] * actuator_force[a, i_b]`.
- **Impact:** Native 10,000+ FPS GPU musculoskeletal + soft-body MPM simulation in Genesis (~280 LOC).

---

## 2. Upstream Proposals for `embodied-ai-nexus/Eden`

1. **Decouple Pure-Python / CPU-MuJoCo Subpackages from Eager `import genesis`:**
   - *Problem:* `eden/__init__.py` and several subpackage `__init__.py` modules eagerly `import genesis`, preventing lightweight utilities (`eden/utils/chumpy_pickle.py`, `eden/extensions/pose_estimation/gvhmr/convert.py`, `eden/extensions/retargeting/flashdexretarget/ik.py`, `eden/managers/catalog.py`, and `eden/learning/wrappers/rsl_rl/*`) from being imported in MuJoCo-Warp / `mjlab` environments or on Python versions without `genesis-world` wheels.
   - *Fix:* Guard `import genesis as gs` behind `TYPE_CHECKING` or function-scope lazy imports in utility/learning modules (mirroring PEP 562 lazy exports in `myosuite`).

2. **Replace `nlopt` SLSQP in `dex_retargeter` with Analytic-Jacobian `scipy.optimize.minimize`:**
   - *Problem:* `eden/extensions/retargeting/dex_retargeter/optimizer.py` requires `nlopt` (which frequently lacks wheels on newer Python / Windows / ARM64 targets) and builds kinematic frames via Genesis entities.
   - *Fix:* Upstream the `scipy.optimize.minimize(..., method="SLSQP", jac=True)` + `mujoco.mj_jacSite` + `JointCouplings` implementation from [`myosuite/biomechanics/hand_retarget.py`](../../myosuite/biomechanics/hand_retarget.py) as a zero-`nlopt`, zero-GPU backend option.

3. **Fix `FrameNormalizer` Double-Normalization Note & Add Multi-Format Clip Support in `AmpMotionSampler`:**
   - Upstream [`training/eden_rl/eden_rl/amp.py`](../../training/eden_rl/eden_rl/amp.py)'s support for `qpos`-based `.npz` clips (`qpos_slice`) alongside `motion_<i>` bundles.

4. **Generalize `HIMEstimator` Beyond Fixed 3-D Velocity (`VELOCITY_DIM = 3` -> `target_dim: int = 3`):**
   - *Problem:* `eden/learning/wrappers/rsl_rl/him/estimator.py` hardcodes `VELOCITY_DIM = 3`.
   - *Fix:* Parameterize `target_dim: int = 3` as implemented in [`training/eden_rl/eden_rl/him.py`](../../training/eden_rl/eden_rl/him.py), unlocking privileged latent estimation for actuator temperature, motor wear, payload mass, and musculoskeletal fatigue (`3CC-r`).

5. **Add `MuscleActuationActionTerm` & `myoHand` / `myoArm` / `myoFullBody` Presets:**
   - Register an `ActionTerm` wrapping `GenesisMuscleBridge` (`control_dofs_force`) and add `myoHand` (`up=(0,-1,0), front=(0,0,-1)` in `lunate_r`) to `dex_retargeter/configs/` and `otretarget`.

---

## 3. Upstream Proposals for `MyoHub/myo_sim`

1. **Promote `myosuite/physics/hill_muscle.py` into `myo_sim.physics.hill_muscle`:**
   - `myo_sim.physics` already hosts `fatigue.py` (3CC-r) and `joint_coupling.py`. Moving [`myosuite/physics/hill_muscle.py`](../../myosuite/physics/hill_muscle.py) (`MuscleParams`, `muscle_activation_dot`, `step_muscle_activation`, `muscle_gain`, `muscle_bias`, `HillMuscleModel`, `strip_muscles_for_joint_torque_sim`) into `myo_sim.physics` makes exact NumPy/PyTorch Hill dynamics available to every downstream simulator (`MyoSuite`, `Genesis`, `IsaacLab`, `SuperMimic`) from a single source of truth.

2. **Promote Differentiable Analytical Tendon Geometry into `myo_sim.physics.tendon_geometry`:**
   - Generalize [`training/supermimic/supermimic/muscle_geometry.py`](../../training/supermimic/supermimic/muscle_geometry.py) from `myoFullBody`-specific `SkeletonKinematics` to arbitrary `MjModel` kinematic trees (handling hinge, slide, ball, free, and `JointCouplings`) inside `myo_sim.physics`, computing analytical `(L, L_dot, J_moment)` on PyTorch GPU without autograd overhead when desired.

3. **Standardize Upper-Limb Joint & Site Naming Across `hand`, `arm`, and `myofullbody` Fragments:**
   - *Finding during implementation:*
     - `ModelBuilder().attach_fragment("hand")` uses `_r` suffixes (`lunate_r`, `S_grasp_r`, `THtip_r`..`LFtip_r`, and orders `cmc_flexion_r` before `cmc_abduction_r`).
     - `ModelBuilder().attach_fragment("arm")` omits `_r` suffixes (`lunate`, `S_grasp`, `THtip`..`LFtip`, and orders `cmc_abduction` before `cmc_flexion`).
   - [`myosuite/biomechanics/hand_retarget.py`](../../myosuite/biomechanics/hand_retarget.py) handles both via `MYOHAND_PRESET` and `MYOARM_HAND_PRESET` (`preset_for_model`), but standardizing `<site>` aliases (`wrist_r`, `palm_r`, `THtip_r`..`LFtip_r`) and consistent joint ordering in `myo_sim` `0.3.x` will simplify cross-fragment retargeting and policy transfer.

4. **Ship First-Class `myo_sim.build.build_joint_torque_spec(fragment)` Recipe:**
   - Expose `strip_muscles_for_joint_torque_sim` directly from `myo_sim` so torque-actuated baselines and non-muscle rigid solvers (`Genesis`, `Pinocchio`, `Drake`) can instantiate inertial/kinematic twins of every MyoSuite model in one call.

---

## 4. Upstream Proposals for `ttktjmt/mjswan`

1. **Ingest `<stem>.policy_spec.json` Sidecars Emitted by `myosuite-export-onnx --policy-spec`:**
   - [`myosuite/utils/export_onnx.py`](../../myosuite/utils/export_onnx.py) now writes schema-versioned `.policy_spec.json` files declaring input/output tensor names, shapes, recurrent state pairs (`h_in`/`h_out`, `c_in`/`c_out`), and SHA-256 checkpoint provenance. `mjswan` can read this sidecar to auto-configure `onnxruntime-web` feeds without manual shape configuration.
2. **Thread Recurrent Hidden States (`h_in` / `c_in`) Across Browser Control Steps:**
   - Match `verify_onnx_on_cpu` in `export_onnx.py`: allocate zero `(num_layers, 1, hidden_dim)` `Float32Array` buffers on episode reset and feed `h_out -> h_in` (`c_out -> c_in`) on every control tick so GRU/LSTM motor-adaptation policies run in the browser.
