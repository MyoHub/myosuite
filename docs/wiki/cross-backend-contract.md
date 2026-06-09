# Cross-Backend Contract

**Read this before adding mjlab support to a task that will be evaluated on CPU, MJX, or browser (mjswan).**

A policy is only portable across backends if all five invariants below hold. Violating any produces wrong behaviour at eval time without an obvious error.

---

## The Five Invariants

### 1. Observation vector — identical shape, order, and units

- Obs terms registered in the same order in `ObservationGroupCfg` (mjlab) and the browser config.
- Per-term `scale` and `clip` must match exactly — these are applied in TypeScript at inference.
- **No `VecNormalize` or running mean/std.** The browser runtime cannot load normalization statistics. Express normalization as fixed `scale` in `ObservationTermCfg`.

### 2. Action space — identical dimensionality, scaling, and activation

| Training class | Browser equivalent | Scaling |
|---|---|---|
| `MyoMuscleActivationAction` | `MuscleActionCfg` | `sigmoid(x)` → ctrl ∈ [0, 1] |
| `TendonLengthActionCfg` | `JointPositionActionCfg` | `scale × x + default_pos` |

`encoder_bias` in the exported JSON must equal `model.key_qpos[0]` — ensure the XML has a keyframe at index 0 representing the neutral pose.

### 3. Control timing — identical `ctrl_dt`

```
ctrl_dt = decimation × sim_dt
```

mjswan hardcodes decimation as `round(0.02 / model.opt.timestep)` (50 Hz control). Use only timesteps that are integer divisors of 0.02:

| `timestep` | Browser decimation | ctrl_dt |
|---|---|---|
| 0.002 s | 10 | 0.02 s ✓ |
| 0.001 s | 20 | 0.02 s ✓ |
| 0.003 s | 7 | 0.021 s ✗ |

### 4. Model XML — identical physics

- Use `myo_sim.get_path(...)` in the mjlab `spec_fn` — never hardcode paths.
- The browser model must come from the same source XML used during training.
- Do not strip actuators, sensors, or meshes for deployment.
- If DR is applied during training (mass, friction), the browser uses the nominal model — policies must be robust to this by design.

### 5. Observation normalization — per-term scale/clip only

Fixed `scale` in `ObservationTermCfg` is exported to JSON and replicated in TypeScript. `VecNormalize` stats are not.

---

## Export Checklist

- [ ] Obs term key order identical between mjlab and browser config (assert in `test_mjlab_task_builder.py`)
- [ ] Obs term `scale` and `clip` values match exactly
- [ ] No running mean/std normalization
- [ ] `ctrl_dt = decimation × timestep = 0.02 s`
- [ ] XML path via `myo_sim.get_path(...)`
- [ ] Keyframe at index 0 for `encoder_bias`
- [ ] Action class exported correctly
- [ ] Smoke-test rollout: 10 steps in browser without NaN

---

## Parity Status

| Task | gym ↔ mjlab obs | gym ↔ mjlab action | mjswan export | Blocker |
|---|---|---|---|---|
| Elbow | ✓ 9D | ✓ 6D sigmoid | ✗ | No TypeScript: `pose_err`, `act`, `qvel×ctrl_dt` |
| Walk | ✓ 403D | ✓ 80D sigmoid | ✗ | No TypeScript: all 12 custom obs terms |
| TableTennis | ~ | ~ | ✗ | Closure-based obs not introspectable |
| Saber | – | – | – | Not implemented |

All passing parity tests live in `myosuite/tests/test_mjlab_task_builder.py`.

**To unblock mjswan export:** either contribute TypeScript obs term implementations to the [mjswan repo](https://github.com/ttktjmt/mjswan), or rewrite mjlab obs functions to use only mjswan built-ins (`joint_pos_rel`, `joint_vel_rel`, etc.) where semantically equivalent.
