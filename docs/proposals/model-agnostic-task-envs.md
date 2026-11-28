# RFC: Model-Agnostic Musculoskeletal Task Envs & Separating Task Contracts from Learning Recipes

## 1. Motivation & Root-Cause Findings (`myo_sim 0.2.5` vs `0.3.0`)

Attempting to train or evaluate `myo_sim 0.3.0` (`v1`: `myofullbody`, `myofullbody_simple`, `myolegs`) inside existing MyoSuite environments exposed three concrete architectural couplings between **task definitions**, **musculoskeletal plant versions**, and **RL shaping choices**:

1. **Runtime Plant Coupling (`mj_step` vs. `myo_sim.compliant` / `myo_sim.warp`):**
   - `myo_sim 0.3.0` models default to **series-elastic compliant tendons** (`KIND_DGF = 1.0`, `userdata` fiber states) and universal De Groote–Fregly curves (`KIND_DGF_RIGID = 2.0`).
   - Stepping a `v1` compliant model with raw `mujoco.mj_step` (CPU) or raw `mujoco_warp.put_model` (GPU `mjlab`) ignores series tendon stretch (`L_fiber = L_path - L_slack`), producing **up to `1,202 N` (`+110%`) active/passive force errors** (`180 / 196` actuators wrong on `myofullbody_simple` at `qpos0`) and multi-kN passive exponential spikes at flexion extremes.
2. **Hardcoded Kinematic & Actuator Topology Assumptions (`v0` vs. `v1`):**
   - `v1` models introduce **spherical `mjJNT_BALL` acromioclavicular joints** (`nq != nv`, unlimited quaternion coordinates), **mixed tendon + joint transmissions** (`iap_lumbar_r/l` `mjTRN_JOINT`), **revised 33-DoF/47-muscle hand subtrees** (`cmc_pronation` equalities, `node_*` tendon ties, spatial ligaments), and **floating 12-DoF scapulothoracic girdles**.
   - Existing `MimicEnv` (`ModelInfo`, `load_clip`, `build_native_mimic_fullbody_spec`), `mjlab` (`init_state_from_model`, `MyoMuscleActivationAction`), and `LegWalkEnvV0` assumed scalar-only `nq == nv` hinges, `100%` `mjTRN_TENDON` actuators, and exact `v0` clip joint lists.
3. **Method- & Model-Dependent Shaping Baked Into Task Environments:**
   - Standard `v0` locomotion rewards (`vel_reward - 100 * done - w_act * ||act||^2` with independent per-muscle `[-1, 1]` actions) were tuned on `61 N/cm^2` welded-trunk `myolegs` (`20` independent DoFs, `80` rigid muscles).
   - On `v1` whole-body models (`36 N/cm^2` Handsfield specific tension, `49–118` independent DoFs, `172–558` muscles), independent Gaussian exploration co-contracts antagonists while large fall (`-100`) and unnormalized effort penalties trap PPO in a standing local minimum—whereas **anatomical synergy + residual action parameterization**, **`v1`-calibrated reward weights** (or DeepMimic/AMP shaping + RSI), and **optional scapulothoracic locking (`keep_scapula_dofs=False`)** train cleanly across walking, running, jumping, and agility tasks.

---

## 2. Design Question: Should Task Envs Define Dense Rewards at All?

There are two complementary perspectives:

| Option | Task Env Responsibility | Reward Ownership | Pros & Trade-offs |
|---|---|---|---|
| **Option A — Pure Task Envs (`reward_mode="task_only"`)** | Scene, goal/command, observations, ground-truth `solved` (success) and `terminated` (failure), plus diagnostic `task_metrics` in `info`. | Owned entirely by the **learning method** (`(task, model, algo)` wrapper/runner: PPO shaping, DeepMimic tracking, AMP style reward, Dreamer imagination, etc.). | Cleanest scientific boundary: two methods (e.g. PPO with velocity shaping vs. AMP vs. trajectory DAgger) are compared solely on the **identical `solved` / `terminated` task metric**, never conflating shaping return with task success. *Trade-off:* out-of-the-box `make_env(id)` with generic SB3/RSL-RL fails on sparse `0/1` musculoskeletal tasks unless paired with a shaping wrapper/preset. |
| **Option B — Model-Aware Default Shaping + `task_only` Opt-In (Implemented on `myosim_v1_experimental`)** | Same model-invariant scene, observations, and ground-truth `solved` / `terminated` contract, **plus** `reward_mode: Literal["default", "task_only"] = "default"`. | Default shaping is resolved via `resolve_training_recipe(task_family, model, method)`: `v0` models keep exact legacy HF-baseline weights (`100%` checkpoint parity); `v1` models use `v1`-calibrated weights. Setting `reward_mode="task_only"` (or `TaskOnlyRewardWrapper`) strips all shaping cleanly. | Preserves `make_env(id)` usability for quick SB3/mjlab baselines on both `v0` and `v1` while giving method authors a first-class, zero-shaping `task_only` contract and reusable `(task, model, method)` `TrainingRecipe` configs. |

### Recommendation for Discussion
1. **Standardize Ground-Truth Task Outcomes First:** Every MyoSuite task must expose unambiguous, model-invariant `info["solved"]`, `terminated` (physical failure), and `info["task_metrics"]` (e.g. velocity error m/s, waypoint distance m, tracking MPJPE mm), which `myosuite_training.evaluation` scores identically across all methods.
2. **Support `reward_mode="task_only"` Universally, Keep `reward_mode="default"` via `TrainingRecipe`:** Rather than breaking standard Gymnasium `PPO("MlpPolicy", make_env(...))` users, factor default shaping weights, action structure (`buchanan` vs. `SynergyResidualActionWrapper`), scapulothoracic locking, and runner hyperparameters into `myosuite.core.training_recipes.TrainingRecipe` keyed by `(task_family, model_generation, method)`.

---

## 3. Summary of Prototype on `myosim_v1_experimental`

1. **`myosuite/physics/compliant_plant.py`:** Automatic `v0` vs. `v1_compliant` / `v1_rigid` detection, `prepare_spec_for_plant` (`userdata` sizing), `CpuCompliantPlant` (`MyoGymnasiumEnv._step_physics` & `_refresh_physics_after_reset`), and `install_mjlab_compliant_hook` (`myo_sim.warp.put_model` inside `mjlab.sim.Simulation`).
2. **Topology- & Generation-Agnostic `MimicEnv`, `WaypointEnv`, and `LegWalkEnvV0`:**
   - Supports `mjJNT_BALL` (`nq != nv`) and mixed `mjTRN_TENDON` + `mjTRN_JOINT` (`iap_lumbar`) actuators on both CPU and `mjlab`.
   - Registers `"myofullbody"`, `"myofullbody_rigid"`, `"myofullbody_simple"`, `"myofullbody_simple_locked_st"` (`172M`, locked neutral scapulae), and `"myofullbody_simple_rigid"`, plus `"myolegs_v1"`.
   - Allows `v0`-retargeted clips to load directly on `v1` models by defaulting newly added anatomical stabilizer DoFs to `qpos0`.
3. **`myosuite/core/training_recipes.py` & `myosuite/envs/synergy_wrapper.py`:**
   - Model-agnostic `build_anatomical_synergy_map(model)` + `SynergyResidualActionWrapper` + `TaskOnlyRewardWrapper` + `resolve_training_recipe(task, model, method)`.
