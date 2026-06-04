# Cross-Backend Contract

**Read this before adding mjlab support to a task that will run in CPU eval, MJX, or
mujoco_wasm (browser via mjswan).**

A policy trained on one backend is only portable to another if every invariant in
this document is satisfied. Violating any of them silently produces wrong behaviour
at eval time — the policy runs but the obs or actions mean something different.

---

## Why this matters: the mjswan deployment path

[mjswan](https://github.com/ttktjmt/mjswan) packages a trained policy as an **ONNX
file** and runs it inside a browser Web Worker alongside the same MuJoCo model XML
used during training. The browser runtime:

1. Steps the MuJoCo physics `decimation` times per policy call (computed as
   `round(0.02 / model.opt.timestep)` — hardcoded 50 Hz control frequency).
2. Assembles the observation vector by calling each registered obs term in
   **registration order** and concatenating results.
3. Applies per-term `scale` and `clip` in TypeScript — **no running mean/std**.
4. Feeds the concatenated vector into the ONNX model.
5. Maps policy outputs to MuJoCo `ctrl` via `scale * action + offset + encoder_bias`
   (where `encoder_bias` is the model's default joint position).

If any of these five steps differs from what happened during mjlab training, the
policy will receive wrong inputs or produce wrong control signals.

---

## The Five Invariants

### 1. Observation vector: identical shape, order, and units

The obs vector fed to the policy at mjlab training time must be byte-for-byte
identical to what the browser runtime assembles.

**Requirements:**

- Every obs term must be registered in the same order in both the mjlab
  `ObservationGroupCfg` and the mjswan `ObservationGroupCfg`.
- Per-term `scale` and `clip` values must match exactly — these are applied at
  inference time in TypeScript, so any normalization done *inside* the mjlab obs
  function must be moved to the `scale`/`clip` fields instead.
- **Do not use running mean/std normalization** (`VecNormalize` or equivalent) when
  training a policy that will be exported for mjswan. The browser runtime has no
  mechanism to apply or load normalization statistics. If normalization is needed,
  embed it as fixed `scale` terms in the obs group config.
- Units and coordinate frames must match: `entity.data.joint_pos` (radians, relative
  to default pose if `subtractDefault=True`) vs whatever the mjlab obs term returns.

**How to verify:**

```python
# Serialize obs config to JSON and diff training vs export
import json
train_obs = [list(group.terms.keys()) for group in env_cfg.observations.values()]
mjswan_obs = scene.observation_group.to_dict()
assert train_obs == mjswan_obs["term_order"], "obs term order mismatch"
```

Add an assertion test in `test_mjlab_task_builder.py` that confirms obs term key
order is stable across the training config and the mjswan export config.

---

### 2. Action space: identical dimensionality, scaling, and activation function

The raw policy output vector must map to MuJoCo `ctrl` by the same transform at
both training time (mjlab) and inference time (browser).

**For muscle-tendon tasks (elbow, walk, soccer, chasetag):**

mjlab training uses `MyoMuscleActivationAction`, which applies
`sigmoid(raw_action) → [0, 1]` before writing to tendon ctrl. The browser must
apply the same sigmoid.

mjswan's `MuscleActionCfg` handles this — it detects `MyoMuscleActivationAction`
via the adapter and sets a `sigmoid` flag in the JSON config. Confirm this is wired
correctly when exporting.

| Training action class | Browser equivalent | Scaling |
|---|---|---|
| `MyoMuscleActivationAction` | `MuscleActionCfg` | `sigmoid(x)` → ctrl ∈ [0, 1] |
| `TendonLengthActionCfg` | `JointPositionActionCfg` | `scale * x + default_pos` |
| Mixed (muscles + joints) | `MixedActionCfg` | per-actuator sigmoid or linear |

**Requirement:** The `encoder_bias` in the exported JSON must equal the model's
`qpos_default` for the actuated joints. mjswan computes this automatically from
`model.key_qpos[0]` (the first keyframe) — make sure the training XML has a
keyframe at index 0 that represents the desired neutral pose.

---

### 3. Control timing: identical `ctrl_dt`

The policy was trained to receive observations and output actions at a specific
control frequency. If the browser runs at a different frequency, the policy's
implicit temporal model is wrong.

**Requirement:**

```
ctrl_dt = decimation × sim_dt
```

must be identical in training and deployment. mjswan hardcodes its decimation as
`round(0.02 / model.opt.timestep)`, which gives **50 Hz** for any timestep that
divides 0.02 evenly.

| `model.opt.timestep` | Browser decimation | ctrl_dt |
|---|---|---|
| 0.002 s (default) | 10 | 0.02 s = 50 Hz ✓ |
| 0.001 s | 20 | 0.02 s = 50 Hz ✓ |
| 0.004 s | 5 | 0.02 s = 50 Hz ✓ |
| 0.003 s | 7 | 0.021 s ≠ 50 Hz ✗ |

Use only timesteps that are integer divisors of 0.02. The `mjlab_env_cfg_from_task_config`
factory computes `episode_length_s` from the actual `sim_cfg.mujoco.timestep` (fixed
in commit `2f0be47`) — verify the training `decimation` also matches.

**In the mjlab config:**

```python
return mjlab_env_cfg_from_task_config(
    ...,
    decimation=10,                                          # ctrl_dt = 0.002 × 10 = 0.02s
    sim_cfg=SimulationCfg(mujoco=MujocoCfg(timestep=0.002)),
)
```

**In the mjswan export:** no extra config needed — the browser reads `timestep`
directly from the `.mjb`/`.mjcf` model and derives decimation automatically.

---

### 4. Model XML: identical physics

The MuJoCo model run in the browser must be the same file used to generate training
trajectories. mjswan serialises the model as `.mjb` (compiled binary) or `.mjz`
(MjSpec) at build time. Either format must come from the same source XML.

**Requirements:**

- Use `myo_sim.get_path(...)` in the mjlab `spec_fn` — never hardcode paths.
- The mjswan scene config must point to the same resolved XML path.
- Do not strip actuators, sensors, or meshes from the XML for deployment — the
  browser needs the full model to replicate training dynamics.
- If the task applies domain randomization (mass, friction) during training, the
  browser uses the nominal model. Policies should be robust to this by design
  (train with DR on mjlab, eval on nominal). Document the DR range in the task config.

---

### 5. Observation normalization: per-term scale/clip only

mjswan's TypeScript runtime applies per-term `scale` and `clip` but **cannot** load
or apply running mean/std statistics. This rules out post-hoc `VecNormalize`.

**Rule:** If an obs term needs normalization, it must be expressed as a fixed `scale`
multiplier in the `ObservationTermCfg`, not as a wrapper applied after training.

```python
# CORRECT — fixed scale baked into the term config, exported to JSON
ObservationTermCfg(
    func=_walk_obs_qvel,
    scale=0.1,          # exported, replicated in TypeScript
)

# WRONG — VecNormalize wraps the env, stats not exportable
env = VecNormalize(env, norm_obs=True)
```

If you inherited a policy trained with `VecNormalize`, the running mean/std must be
exported separately and inserted as fixed scale/bias in the obs term config before
re-exporting the ONNX.

---

## Export Checklist

Before marking a task as `beta` in the backend matrix, verify all five invariants:

- [ ] Obs term **key order** is identical between mjlab `ObservationGroupCfg` and
      mjswan `ObservationGroupCfg` (assert in `test_mjlab_task_builder.py`).
- [ ] Obs term **scale and clip** values match exactly (assert per-term).
- [ ] **No running mean/std** normalization — fixed `scale` only.
- [ ] `ctrl_dt = decimation × timestep` equals `0.02 / round(0.02 / timestep)`.
- [ ] Model XML path resolves via `myo_sim.get_path(...)`.
- [ ] A keyframe at index 0 exists for `encoder_bias` (default joint positions).
- [ ] Action class exported correctly (`MuscleActionCfg` for sigmoid, `JointPositionActionCfg` for linear).
- [ ] A smoke-test rollout runs 10 steps in the browser without NaN or divergence.

---

## Parity test pattern: CPU ↔ mjlab obs/action dim

Add to `myosuite/tests/test_mjlab_task_builder.py` for each registered task:

```python
def test_walk_obs_action_dim_parity():
    """Obs and action dims must match between CPU gym env and mjlab config."""
    pytest.importorskip("mjlab")
    import gymnasium as gym
    import myosuite
    myosuite.register_all_envs()

    cpu_env = gym.make("myoLegWalk-v0")
    mjlab_cfg = _make_walk_env_cfg()

    cpu_obs_dim = cpu_env.observation_space.shape[0]
    mjlab_obs_dim = sum(
        term.func(None).__len__()   # or use a mock env to get term dim
        for term in mjlab_cfg.observations["policy"].terms.values()
    )
    assert cpu_obs_dim == mjlab_obs_dim, (
        f"obs dim mismatch: CPU={cpu_obs_dim} mjlab={mjlab_obs_dim}"
    )

    cpu_act_dim = cpu_env.action_space.shape[0]
    mjlab_act_dim = len(mjlab_cfg.actions["muscles"].actuator_names)
    assert cpu_act_dim == mjlab_act_dim
```

The exact mechanism for extracting mjlab obs dim varies by term — use a minimal
`ManagerBasedRlEnv(cfg)` instantiation if needed (requires GPU).

---

## Status: gym (CPU) ↔ mjlab parity

| Task | Obs parity (gym ↔ mjlab) | Action parity | Tests |
|---|---|---|---|
| Elbow (`myoElbowPose1D6MFixed-v0`) | ✓ 4 terms, 9D: `[qpos, qvel, pose_err, act]` | ✓ `MyoMuscleActivationActionCfg` sigmoid, 6D | `test_elbow_obs_keys_match_cpu`, `test_elbow_action_dim_matches_cpu` |
| Walk (`myoLegWalk-v0`) | ✓ 12 terms, 403D: exact `DEFAULT_OBS_KEYS` order | ✓ `MyoMuscleActivationActionCfg` sigmoid, 80D | `test_walk_obs_keys_match_cpu`, `test_walk_action_dim_matches_cpu` |
| TableTennis | ~ closure-based obs, partial | ~ | No parity tests yet |
| Saber | Not implemented (separate entity file needed) | – | – |

All passing tests live in `myosuite/tests/test_mjlab_task_builder.py`.

## Known gaps: mjlab → mjswan (browser) export

The gym↔mjlab parity is complete for elbow and walk. The **mjswan export path** is
still blocked because mjswan's TypeScript runtime can only compute the following
standard obs functions:

```
base_ang_vel, base_lin_vel, builtin_sensor, generated_commands,
height_scan, joint_pos_rel, joint_vel_rel, last_action, projected_gravity
```

Our custom obs functions — `_elbow_obs_qpos`, `_elbow_obs_qvel`, `_elbow_obs_pose_err`,
`_elbow_obs_act`, `_walk_obs_qpos_without_xy`, `_walk_obs_com_vel`,
`_walk_obs_feet_heights`, `_walk_obs_feet_rel_positions`, etc. — have **no TypeScript
counterparts in mjswan**. The browser cannot reproduce these obs calculations, so
exporting an ONNX policy from mjlab training would produce wrong inputs at inference.

| Task | mjswan export | Blocking gap |
|---|---|---|
| Elbow | ✗ blocked | `pose_err` (fixed-target subtraction), `act` (MuJoCo `data.act`), `qvel×ctrl_dt` scaling |
| Walk | ✗ blocked | All 12 custom terms need TypeScript implementations in mjswan |
| TableTennis | ✗ blocked | Closure-based obs not introspectable |
| Saber | – | Not implemented |

**To unblock mjswan export** for elbow or walk, one of:
1. Contribute TypeScript obs term implementations to the [mjswan repo](https://github.com/ttktjmt/mjswan) for each custom function.
2. Rewrite the mjlab obs functions to use only the built-in mjlab mdp functions above, if semantically equivalent (possible for `qpos`→`joint_pos_rel`, `qvel`→`joint_vel_rel`; not possible for `pose_err`, `act`, walk-specific terms).

This is tracked as a known future work item. Do not mark elbow/walk as "mjswan beta" until at least one of the above paths is complete and a smoke-test rollout passes in the browser.

---

## See also

- `docs/wiki/mjlab-design-guide.md` — mjlab entity/state/DR patterns
- `docs/wiki/writing-term-functions.md` — CPU/MJX backend-agnostic term functions
- `docs/wiki/engineering-standards.md` — full task checklist
- [mjswan source](https://github.com/ttktjmt/mjswan) — browser runtime and ONNX export
