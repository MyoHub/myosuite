# Issue #42 / #45 / #56 — mjlab saber rewrite plan

**Direction change (2026-05-28)**:
- Do **not** keep or reintroduce a MuJoCo stepping fallback in the mjlab saber task.
- The split-scene mjlab task may retain a small native dynamics mismatch versus the CPU monolithic task for now.
- The priority is a **clean mjlab-native implementation** with as much shared reset / step / reward / observation logic as possible.
- Prefer rewriting the saber mjlab registration into a cleaner file named `register_mjlab_saber.py`.
- Remove every non-essential branch, especially all `mimic_mix`-specific code from the default saber mjlab path.

## What is already done

- [x] Standalone XML assets are in use for the mjlab saber scene:
  - `myoarm_saber_body.xml`
  - `left_lightsaber.xml`
  - `right_lightsaber.xml`
  - `saber_targets.xml`
- [x] `saber_p0_mjlab_env.py` constructs separate robot / left saber / right saber entities.
- [x] The target pool no longer writes directly into `env.sim.wp_data.mocap_pos`.
- [x] Saber reset writes saber freejoint state through `write_root_state_to_sim()`.
- [x] Target motion writes per-target mocap poses through `write_mocap_pose_to_sim()`.
- [x] Runtime ids are rebuilt from entity indexing instead of reusing monolithic-model ids.
- [x] `_build_saber_p0_mjmodel_and_spec()` / mimic helpers compile from the split XML assets.
- [x] Saber parity tests no longer read target positions from `env.sim.wp_data.mocap_pos`.

## Remaining work from issue #42

- [x] Remove `_install_saber_mujoco_step_fallback` and any related MuJoCo-step workaround code.
- [ ] Rebaseline saber mjlab validation around the clean split-scene behavior after the fallback is removed.
- [x] Document the per-target-entity path: dynamics parity documented in `docs/saber_backend_alignment/KNOWN_DIFF.md` (PR feat/saber-scene-cfg-consolidation). `test_myo_challenge_saber_mjlab_matches_cpu_control_contract` now passes.
- [x] NEW: CPU path now uses `saber_scene_callable` from `saber_scene_assets.py`; both the CPU `ModularTaskEnv` and the mjlab backend share the same XML-asset scene assembly. `_add_saber_target_pool_scene` is deprecated and has been removed. (PR feat/saber-scene-cfg-consolidation)

## Remaining work from issue #45

- [x] Remove the bespoke `SaberTaskEntityCfg` / `SaberTaskEntity` path if it is no longer needed for task logic.
- [x] Port `SaberTaskLogicState` toward the canonical mjlab `ManagerTermBase` pattern instead of `id(env)` / ad-hoc env patching.
- [x] Keep reset logic in `EventTermCfg`, reward logic in `RewardTermCfg`, observation logic in `ObservationTermCfg`, and fall / done logic in `TerminationTermCfg`.
- [x] Preserve the existing shared reward / observation / reset semantics while moving state ownership into manager terms.
- [x] Reassess whether target-hit detection can use a canonical contact sensor path without regressing current behavior. (See `docs/saber_backend_alignment/KNOWN_DIFF.md` — ContactSensor assessed; `wp_data.contact` used instead for body/helmet recycling.)

## Remaining work from issue #56

- [x] Replace raw `task_cfg.reward.extra[...]` / `.get(...)` accesses with typed dataclass fields.
- [x] Update `SaberP0TaskReward` / `SaberP0Task` config wiring so saber env code does not depend on untyped reward-extra dicts.
- [x] Update tests that currently assert against `reward.extra[...]` dict keys. (`modular_env.py` now reads from `task_config.saber_cfg`; parity test updated to use `obs.extra["num_visible"]` directly.)

## Additional cleanup / regression surfaces

- [x] Restore the `mj_model` reference path expected by sections 2 and 3 of `tutorials/mc26/mc26_PyTorch_vs_mjlab_Policies.ipynb`.
- [x] Fix visualisation/sync issues introduced by switching to `write_mocap_pose_to_sim`.
- [x] Move the default saber mjlab registration out of `saber_p0_mjlab_env.py` into a cleaner `register_mjlab_saber.py`.
- [x] Migrate all internal imports/tests to `register_mjlab_saber.py` in the same change; do not keep a compatibility shim. Keep this registration config as small as possible, e.g., by introducing reusable util methods in helper files (ideally used by both the CPU and the mjlab envs).
- [x] Remove saber-specific code from `mimic_mjlab_env.py`; route saber-mimic registration through `register_saber_p0_mjlab_task_with_mimic_mix` in `register_mjlab_saber.py` so saber-mimic uses the same split-scene terms as `myoChallengeSaberP0-v0`. (PR feat/saber-mimic-cleanup)

## Immediate execution order

1. Restore the notebook `mj_model` access path.
2. Introduce typed saber reward/obs config dataclasses for issue #56.
3. Rewrite the default saber mjlab registration into a shorter cleaner standalone module.
4. Migrate all imports/tests to the new module in the same change.
5. Re-run saber-specific and relevant repo validation.

---

# MuscleMimic mjlab Library — Implementation Plan

**Goal**: Replicate full MuscleMimic training (https://github.com/amathislab/musclemimic) in myosuite via the mjlab backend. The existing `mimic_mjlab_env.py` covers random-target and basic clip-tracking; this plan fills the gaps for end-to-end motion imitation training.

**Reference algorithm**: PPO (PureJaxRL-style) + Muon optimizer, ~2B timesteps, 8192 parallel envs, DeepMimic-style reward, 5-step lookahead obs, AMASS/KIT retargeted clips.

---

## Gap Analysis (current state vs MuscleMimic)

| Feature | Current | Missing |
|---|---|---|
| Random-target mjlab env | ✅ `mimic_mjlab_env.py` | — |
| Clip-tracking mjlab env | ✅ ClipTrajectorySource + phase obs | — |
| **DeepMimic reward** (multi-term weighted) | ❌ single `exp(-20·err)` | site(0.6) + jpos(0.1) + jvel(0.1) + root kinematics |
| **5-step lookahead obs** | ❌ only current-frame ref | lookahead at stride 20 over clip |
| **Early termination** (deviation threshold) | ❌ no physics termination | mean site err > 1m or root threshold |
| **Obs normalizer** (RunningMeanStd) | ❌ raw obs | Welford online normalizer term |
| **AMASS/KIT data pipeline** | ⚠️ sandbox only | promoted to `myosuite/data/` module |
| **Training script** (PPO + Muon) | ❌ | `tutorials/mimic/train_mimic.py` |
| **Validation tests** | ❌ | test_mimic_terms, test_mimic_env, test_mimic_reward_parity |

---

## Phase 1 — DeepMimic Reward Terms  `myosuite/terms/mimic_reward.py`

Pure, backend-agnostic term functions (use `accessor.array_module()` only).

```
mimic_site_tracking_reward(accessor, site_ids, ref_sites, weight=0.6)
    → exp(-20 · mean_euclidean(current_sites - ref_sites))

mimic_joint_pos_reward(accessor, ref_qpos, weight=0.1)
    → exp(-10 · mean_sq(qpos - ref_qpos))

mimic_joint_vel_reward(accessor, ref_qvel, weight=0.1)
    → exp(-1  · mean_sq(qvel - ref_qvel))

mimic_root_pos_reward(accessor, ref_root_pos, weight=0.1)
    → exp(-20 · ||root_pos - ref_root_pos||)

mimic_root_vel_reward(accessor, ref_root_vel, weight=0.1)
    → exp(-1  · ||root_vel - ref_root_vel||)

mimic_root_orient_reward(accessor, ref_root_quat, weight=0.01)
    → exp(-10 · geodesic_angle(root_quat, ref_root_quat))

mimic_composite_reward(accessor, ...) → weighted sum
```

**Validation test**: `myosuite/tests/test_mimic_terms.py`
- Assert each term in [0, 1] for random inputs
- Assert perfect-tracking input → reward ≈ 1.0
- Assert numpy/jax numerical parity (atol 1e-6)

---

## Phase 2 — Lookahead & Termination Terms  `myosuite/terms/mimic_obs.py`

```
mimic_lookahead_obs(accessor, clip_source, k=5, stride=20)
    Returns flattened (k × n_sites × 3) future site positions +
    (k × 3) future root pos deltas + (k × 3) future root vel deltas +
    k phase values. Shape: k*(3*n_sites + 7).

mimic_early_termination(accessor, site_ids, ref_sites, root_err_threshold=0.3,
                         site_err_threshold=1.0) → bool
```

**Validation test**: `myosuite/tests/test_mimic_obs.py`
- Lookahead at t=0 vs t=stride produces shifted-by-one output
- Wraps correctly at clip boundary
- Termination triggers at exactly the threshold value

---

## Phase 3 — Running Mean/Std Normalizer  `myosuite/physics/running_stats.py`

Welford online estimator, backend-agnostic (numpy path + JAX scan path).

```python
@dataclass
class RunningMeanStd:
    mean: array  # (obs_dim,)
    var:  array  # (obs_dim,)
    count: int

def update(stats, batch) -> RunningMeanStd   # pure, returns new stats
def normalize(stats, obs) -> array           # (obs - mean) / sqrt(var + eps)
```

**Validation test**: `myosuite/tests/test_running_stats.py`
- After N standard-normal samples, mean ≈ 0 and std ≈ 1 (atol 0.05)
- JAX scan path matches numpy path

---

## Phase 4 — TaskSpec + Registration  `myosuite/envs/myo/tasks/mimic/specs/mimic_spec.py`

Follow engineering-standards golden rule: one `TaskConfig`, no new class.

```python
@dataclass
class MuscleMimicBimanualTask(TaskConfig):
    model: str = "bimanual_mimic"
    max_episode_steps: int = 300
    obs: ObsSpec = ObsSpec(keys=[
        "joint_pos", "joint_vel", "muscle_len", "muscle_vel",
        "muscle_force", "muscle_excitation", "muscle_act",
        "touch_sensors", "mimic_lookahead",
    ])
    goal: GoalSpec = GoalSpec(target_type="trajectory")
    reward: RewardSpec = RewardSpec(
        terms=["mimic_site", "mimic_jpos", "mimic_jvel",
               "mimic_root_pos", "mimic_root_vel", "mimic_root_orient"],
        weights={"mimic_site": 0.6, "mimic_jpos": 0.1, "mimic_jvel": 0.1,
                 "mimic_root_pos": 0.1, "mimic_root_vel": 0.1, "mimic_root_orient": 0.01},
    )
    actuators: list[ActuatorGroupSpec] = [...]

@dataclass
class MuscleMimicFullBodyTask(TaskConfig):
    ...  # same structure, full-body model

register_task(MuscleMimicBimanualTask(), env_id="myoMimicBimanual-v1",
              backends={"mjlab"})
register_task(MuscleMimicFullBodyTask(),  env_id="myoMimicFullbody-v1",
              backends={"mjlab"})
```

**Validation test**: `myosuite/tests/test_mimic_spec.py`
- `test_registry.py` already covers env_id round-trip; add mimic IDs there
- Smoke-instantiate both envs, call reset() + step(), check 5-tuple shape

---

## Phase 5 — AMASS/KIT Data Pipeline  `myosuite/data/amass.py`

Promote the sandbox retargeting code to a first-class module with a clean API.

```python
class AMASSClipLoader:
    """Load retargeted KIT clips as MotionClip objects."""

    @staticmethod
    def from_hf(model_variant: str, split: str = "train") -> list[MotionClip]:
        """Download pre-retargeted clips from HuggingFace hub."""

    @staticmethod
    def from_npz(path: Path, model_variant: str) -> MotionClip:
        """Load a single retargeted .npz clip."""

    @staticmethod
    def retarget_smpl(
        smpl_path: Path,
        model_variant: str,
        method: str = "gmr",
    ) -> MotionClip:
        """Run full GMR retargeting from raw SMPL-H .npz."""
```

**Validation test**: `myosuite/tests/test_amass_loader.py`
- Load a cached dummy clip (10-frame synthetic), assert `site_xpos.shape` correct
- If HF hub available: download 1 clip, assert non-NaN, assert qpos range sane

---

## Phase 6 — mjlab Env Wiring (extend existing)

Extend `mimic_mjlab_env.py` to wire the new terms into mjlab ManagerBasedRlEnvCfg:

- Replace single-term reward with `_build_deepmimic_reward_cfg()` using Phase 1 terms
- Add lookahead obs term from Phase 2 to `ObservationGroupCfg`
- Add physics termination from Phase 2 to `TerminationTermCfg`
- Add obs normalizer (Phase 3) as a post-processing hook in the obs manager

No new class: extend `register_mimic_mjlab_tasks_with_clip()` signature with
`use_deepmimic_reward: bool = True` and `use_lookahead: bool = True`.

**Validation test**: `myosuite/tests/test_mimic_env_mjlab.py`
- Instantiate `myoMimicBimanual-v1` with mjlab backend + dummy clip
- Run 10 steps, assert reward in (0, 1), assert obs shape matches spec
- Assert early termination fires within 20 steps when agent stands still (zero ctrl)

---

## Phase 7 — Training Script  `tutorials/mimic/train_mimic.py`

Minimal PPO+Muon training entry point (not part of core library, lives in tutorials).

```
python tutorials/mimic/train_mimic.py \
    --env myoMimicFullbody-v1 \
    --clips data/kit_retargeted/ \
    --num_envs 8192 \
    --total_steps 2_000_000_000 \
    --hidden_dim 1024 --n_layers 16
```

Actor-critic: MLP (16×1024, SiLU, LayerNorm) or ResNet variant (flag `--arch resnet`).
Optimizer: Muon (β=0.95, lr=4e-4, linear annealing).
Checkpoint every 50M steps. W&B logging optional.

**Validation**: Unit test `test_mimic_training_step.py` that runs 2 PPO iterations
with 64 envs on CPU (no GPU required) and asserts loss decreases.

---

## Verification Checklist (before PR)

```bash
pre-commit run --all-files
pytest myosuite/tests/test_mimic_terms.py -v          # Phase 1
pytest myosuite/tests/test_mimic_obs.py -v            # Phase 2
pytest myosuite/tests/test_running_stats.py -v        # Phase 3
pytest myosuite/tests/test_mimic_spec.py -v           # Phase 4
pytest myosuite/tests/test_amass_loader.py -v         # Phase 5
pytest myosuite/tests/test_mimic_env_mjlab.py -v      # Phase 6
pytest myosuite/tests/test_mimic_training_step.py -v  # Phase 7
pytest myosuite/tests/test_parity.py -v               # must stay ≤ 1e-7
pytest myosuite/tests/test_registry.py -v
```

---

## File Map

```
myosuite/
├── terms/
│   ├── mimic_reward.py          # Phase 1 — DeepMimic reward terms
│   └── mimic_obs.py             # Phase 2 — lookahead + termination
├── physics/
│   └── running_stats.py         # Phase 3 — Welford normalizer
├── data/
│   └── amass.py                 # Phase 5 — AMASS clip loader
├── envs/myo/
│   ├── tasks/mimic/
│   │   └── specs/mimic_spec.py  # Phase 4 — TaskConfig + registration
│   └── backends/mjlab/
│       └── mimic_mjlab_env.py   # Phase 6 — extended (existing file)
└── tests/
    ├── test_mimic_terms.py
    ├── test_mimic_obs.py
    ├── test_running_stats.py
    ├── test_mimic_spec.py
    ├── test_amass_loader.py
    ├── test_mimic_env_mjlab.py
    └── test_mimic_training_step.py
tutorials/mimic/
└── train_mimic.py               # Phase 7 — training entry point
```

---

## Dependencies

- `mjlab` ≥ 1.3 (already used)
- `optax` with Muon (for training script only)
- `huggingface_hub` (optional, for HF clip download)
- `general_motion_retargeting` (optional, for GMR retargeting)
- `amass` / SMPL-H models (optional, for raw retargeting)

All optional deps guarded by `try/except ImportError` with clear error messages.

---

# feat/saber-mjlab-rebaseline — scope

**Goal**: Rebaseline saber mjlab validation tests after the MuJoCo step fallback removal,
and document the accepted native mjlab dynamics mismatch.

**Steps**:
1. Run `pytest myosuite/tests/test_myo_saber_mjlab_parity.py -v` on a machine with CUDA+mjlab
2. Update snapshot baselines if any tolerance checks fail due to the split-scene / no-fallback path
3. Add a comment block in `test_myo_saber_mjlab_parity.py` (or a `docs/saber_backend_alignment/KNOWN_DIFF.md`) documenting that:
   - Split-scene mjlab saber intentionally has a small residual dynamics difference vs CPU monolithic model
   - The difference is expected and does not block training portability
4. Run full parity suite: `pytest myosuite/tests/test_parity.py -v` (atol ≤ 1e-7 required)

## PR 67 follow-up (2026-06-02)

**Goal**: remove remaining mjlab-only saber obs/reward/termination wrappers from the registration path and reuse shared EnvAccessor-driven term functions wherever possible.

**Audit-first execution order**:
1. Enumerate every callable currently wired by `register_mjlab_saber.py` (including imported saber-specific obs/reward/termination/reset helpers) and map it to the CPU/shared equivalent, if any.
2. For wrappers that already have a shared equivalent in `myosuite/terms/` or `EnvAccessor`, replace the mjlab-specific callable with the shared path.
3. For wrappers that are blocked by missing shared abstractions, decide whether to add those abstractions first (preferred) or leave the logic mjlab-local for now.
4. Only after that decision, refactor the registration and parity tests together and post a PR #67 status comment summarizing what was fixed versus what remains.

---

# feat/saber-term-cfg-refactor — scope (issue #45)

**Goal**: Move saber task logic into canonical mjlab manager terms so that
reset/reward/obs/termination logic lives in the right TermCfg classes.

**Current state**: `SaberTaskLogicState` (ManagerTermBase) holds pool state and
step logic, but the reset fn (`_saber_startup_event`, `_saber_reset_event`,
`_saber_step_event`) and obs fns are wired as EventTermCfg/ObservationTermCfg
already. The remaining gaps:

**Steps**:
1. Audit which parts of `SaberTaskLogicState.__init__` and `step()` belong in
   `EventTermCfg` (reset/step) vs should stay in `ManagerTermBase`
2. Target-hit detection: assess whether `SaberTargetPoolConfig.recycle_contact_geom_substrings`
   can use `ContactSensor` without regressing timing behavior
3. Ensure all state written by `_saber_step_event` is accessible from
   `RewardTermCfg` and `TerminationTermCfg` via the task_state dict (already mostly done)
4. Run: `pytest myosuite/tests/test_saber_env.py myosuite/tests/test_myo_saber_mjlab_parity.py -v`

---

# Procedural hand model: remove myo_sim PR #104 / compose dependency for hand_standard

**Goal**: `hand_standard` (and other hand recipes) build the right-hand model directly
in myosuite from raw myo_sim arm asset XMLs, with no dependency on myo_sim's
compose pipeline (`myo_sim.build.compose.load_right_hand_from_arm_spec`) or the
static `myo_sim/models/hand/myohand_r.xml` artefact from PR #104. Once this lands,
PR #104 (https://github.com/MyoHub/myo_sim/pull/104) can be closed upstream.

## Plan

- [ ] `myosuite/core/xml_compose.py` (new): port `component_children`,
      `expand_component_element`, `build_child_xml_from_components` from
      `myo_sim/build/utils.py` (stdlib `xml.etree.ElementTree` + `copy` + `pathlib` only).
- [ ] `myosuite/core/hand_pruning.py` (new): port from `myo_sim/build/hand.py`:
      `RIGHT_HAND_ACTUATORS`, `RIGHT_HAND_REMOVED_JOINTS`, `HAND_PREVIEW_JOINT_POSE`,
      `add_side_suffix`, `side_name`, `base_actuator_name`, `tendon_wrap_geom_names`,
      quaternion helpers, `apply_joint_pose`, `bake_current_body_poses`,
      `bake_hand_preview_pose`, `prune_arm_spec_to_hand`.
- [ ] `myosuite/core/model_recipes.py`: replace `_myohand_r_path()` / `_hand_builder()`
      with a procedural builder that:
      1. Resolves `myo_sim` asset root via `asset_path_resolver.get_sim_asset_root("myo_sim")`.
      2. Builds the raw right-arm `MjSpec` from
         `arm/assets/myoarm_r_{assets,tendons,muscles,chain}.xml` via the new
         `xml_compose.build_child_xml_from_components` (mirrors
         `load_right_arm_spec()`).
      3. Calls `hand_pruning.prune_arm_spec_to_hand(spec, "r")`.
      4. Wraps via `ModelBuilder().attach_spec(spec, name="hand")`.
      No fallback to static XML or myo_sim compose — procedural path only
      (per user decision 2026-06-18).
- [ ] Update/verify `myosuite/tests/test_model_builder.py` and
      `myosuite/tests/test_hand_recipe_parity.py` still pass — these compare against
      `myo_sim.load("myohand_r")` / `myohand_r.xml` as the **reference**, which is fine
      to keep (myo_sim still ships those for its own users); only myosuite's own
      build path changes.
- [ ] Run full verification suite from CLAUDE.md before considering done.
- [ ] Once merged and parity tests green, close https://github.com/MyoHub/myo_sim/pull/104
      as no-longer-needed by myosuite (confirm with user first).

---

# PyPI regression analysis (myosuite dev branch `mjlab` vs PyPI myosuite 2.12.2)

**Goal**: for every env_id registered in this repo that also exists on PyPI myosuite
2.12.2, produce a numeric regression report covering (a) static model diff
(bodies/joints/tendons/actuators — same method as `tasks/hand_model_pypi_diff_report.md`)
and (b) rollout diff (reset obs + fixed-action trajectory: obs/reward/done arrays).

## Plan
- [ ] Create isolated venv (e.g. `/tmp/myosuite_pypi_venv`), `pip install myosuite==2.12.2`,
      verify it imports independently of this repo's editable install.
- [x] Enumerate env_ids registered in this repo (gymnasium registry, substring `myo`) — ~120 ids found.
- [ ] Enumerate env_ids registered by PyPI myosuite 2.12.2 the same way (separate subprocess
      using the venv python, to avoid import collision with this repo's `myosuite` package name).
- [ ] Intersect the two id lists -> directly comparable set. Record ids unique to each side.
- [ ] For each common env_id:
  - Static: load `MjModel` from both envs' `sim.model`, diff body tree/parents/local pose,
    joint ranges, actuator gainprm/lengthrange, tendon resting length at qpos0.
  - Rollout: `env.reset(seed=0)`, step a fixed deterministic action sequence for ~50 steps,
    record obs/reward/terminated/truncated; compare arrays (max abs diff) between repo env
    and PyPI env of the same env_id.
- [ ] Write `tasks/myosuite_pypi_regression_report.md`: top-line table
      (env_id | static diff severity | rollout obs max-diff | rollout reward max-diff)
      plus narrative on which diffs look like intentional architecture changes
      (e.g. procedural hand model, mjlab backend) vs. unintended numerical regressions.
- [ ] Flag env_ids present only in the repo or only on PyPI.
