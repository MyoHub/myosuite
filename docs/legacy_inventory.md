# MyoSuite4 — Hardcoding & Modularity Gap Analysis

**Engineering Review | April 17, 2026**

---

## Executive Summary

MyoSuite4 has a strong architectural core (term functions, EnvAccessor, TaskConfig) but contains
significant hardcoding debt concentrated in three areas: hardware dispatch logic in `robot.py`,
asset path management across MJX/myobase/mjlab, and environment registration patterns that require
simultaneous edits to 3–4 files per new env. Twelve categories of hardcoding are documented below,
prioritized by blast radius and fix cost.

The highest-ROI interventions are all Small effort: a MODEL_REGISTRY dict, registration factory
functions, and a term registry built from module reflection. None require architectural changes.

---

## Finding 1 — Hardware Dispatch Chains  `robot.py`

**Severity: High | Effort: M | Files: 1**

`robot.py` contains four parallel `if/elif` chains keyed on `device["interface"]["type"]` —
one each in `hardware_init`, `hardware_get_sensors`, `hardware_apply_controls`, and
`hardware_close`. Adding a new hardware backend (e.g. a new motor controller) currently requires
simultaneous edits in all four locations.

```python
# robot.py:155 — hardware_init (repeated pattern)
if device["interface"]["type"] == "dynamixel":   ...
elif device["interface"]["type"] == "optitrack": ...
elif device["interface"]["type"] == "franka":    ...
elif device["interface"]["type"] == "realsense": ...
elif device["interface"]["type"] == "robotiq":   ...
```

The same 5-branch chain appears at lines 155, 244, 299, and 361.

**Recommended pattern:** Abstract `HardwareInterface` base class with `init()`,
`get_sensors()`, `apply_controls()`, `close()` methods; a module-level registry:

```python
HARDWARE_DRIVERS: dict[str, type[HardwareInterface]] = {
    "dynamixel": DynamixelDriver,
    "franka":    FrankaDriver,
    "optitrack": OptitrackDriver,
    "realsense": RealsenseDriver,
    "robotiq":   RobotiqDriver,
}
driver = HARDWARE_DRIVERS[device["interface"]["type"]](**device["interface"])
```

Adding a new device becomes: register one new class, zero existing edits.

---

## Finding 2 — Hardcoded Asset Paths  `mjx/__init__.py`, `myobase/__init__.py`

**Severity: High | Effort: S | Files: 3+**

Asset paths are repeated as string literals across at least three registration files.
Any asset directory reorganization requires a mass-edit hunt.

```python
# mjx/__init__.py:138
_elbow_pose_config["model_path"] = (
    epath.Path(epath.resource_path("myosuite"))
    / "envs/myo/assets/elbow/myoelbow_1dof6muscles.xml"
)
# repeated for finger (line 145), hand pose (line 152),
# finger reach (line 158), hand reach (line 165)
```

`myobase/__init__.py` has 42 `register_env_with_variants()` calls each manually constructing
paths via `str(_ASSETS_ROOT / "elbow" / "myoelbow_1dof6muscles.xml")`.

**Recommended pattern:** A single `MODEL_ASSET_MAP` dict as the source of truth:

```python
_ASSETS = epath.Path(epath.resource_path("myosuite"))
MODEL_ASSET_MAP: dict[str, epath.Path] = {
    "elbow_standard":    _ASSETS / "envs/myo/assets/elbow/myoelbow_1dof6muscles.xml",
    "finger_motor":      _ASSETS / "simhive/myo_sim/finger/motorfinger_v0.xml",
    "finger_myo":        _ASSETS / "simhive/myo_sim/finger/myofinger_v0.xml",
    "hand_pose":         _ASSETS / "envs/myo/assets/hand/myohand_pose.xml",
    "leg_mjx":           _ASSETS / "simhive/myo_sim/leg/myolegs_mjx.xml",
}
```

Then all callers do `model_path=MODEL_ASSET_MAP["elbow_standard"]`. This also aligns
with the existing `ModelBuilder` recipe naming convention — recipes already use string
keys like `"elbow_standard"`.

---

## Finding 3 — MJX Environment Registration Requires 3–4 File Edits

**Severity: High | Effort: M | Files: 3–4**

Adding one new MJX environment currently touches:

1. `mjx/__init__.py` — add config block + branch in `make()` + entry in `ALL_ENVS`
2. `myosuite/tests/test_entry_points.py` — add env name to hardcoded expected set
3. Docs / README stability matrix — manual update

The `make()` function is a 130-line `if/elif` chain (lines 246–380). `ALL_ENVS` at line
173 is a separate list that must stay in sync with `make()` manually.

**Recommended pattern:** Replace the `if/elif` chain with a declarative spec table that
doubles as `ALL_ENVS`:

```python
_MJX_ENV_SPECS: dict[str, tuple[type, Callable[[], ConfigDict]]] = {
    "MjxElbowPoseFixed-v0":   (MjxPoseEnv, lambda: _elbow_fixed_cfg()),
    "MjxElbowPoseRandom-v0":  (MjxPoseEnv, lambda: _elbow_random_cfg()),
    "MjxFingerPoseFixed-v0":  (MjxPoseEnv, lambda: _finger_fixed_cfg()),
    ...
}
ALL_ENVS = list(_MJX_ENV_SPECS)  # derived, never stale

def make(env_name: str):
    if env_name not in _MJX_ENV_SPECS:
        raise ValueError(f"Unknown MJX env {env_name!r}. Available: {ALL_ENVS}")
    cls, cfg_fn = _MJX_ENV_SPECS[env_name]
    registry.register_environment(env_name, cls, cfg_fn)
    return registry.load(env_name)
```

New env → add one line to `_MJX_ENV_SPECS`. `ALL_ENVS` and `make()` update automatically.
Test changes to 0 (test iterates `ALL_ENVS`, not a hardcoded set).

---

## Finding 4 — Copy-Pasted Registration Blocks  `myobase/__init__.py`

**Severity: Medium | Effort: S | Files: 1**

`myobase/__init__.py` contains 42 `register_env_with_variants()` calls. Pose envs and
reach envs each follow an identical template differing only in model path, joint/site
names, and threshold values.

```python
# Lines 110–130 — Elbow pose fixed (abridged)
register_env_with_variants(
    id="myoElbowPose1D6MFixed-v0",
    entry_point="myosuite.envs.myo.myobase.pose_v0:PoseEnvV0",
    max_episode_steps=100,
    kwargs={"model_path": str(_ASSETS_ROOT / "elbow" / "myoelbow_1dof6muscles.xml"),
            "target_jnt_range": {"r_elbow_flex": (2.0, 2.0)},
            "normalize_act": True, "pose_thd": 0.175, ...},
)
# Lines 131–150 — Elbow pose random: identical except target_jnt_range
# Lines 189–208 — Finger motor fixed: identical except model_path + joint names
```

**Recommended pattern:** Factory functions per env family:

```python
def _register_pose_env(env_id, model_key, target_jnt_range, reset_type="random"):
    register_env_with_variants(
        id=env_id,
        entry_point="myosuite.envs.myo.myobase.pose_v0:PoseEnvV0",
        max_episode_steps=100,
        kwargs={"model_path": str(MODEL_ASSET_MAP[model_key]),
                "target_jnt_range": target_jnt_range,
                "normalize_act": True, "pose_thd": 0.175,
                "reset_type": reset_type},
    )

_register_pose_env("myoElbowPose1D6MFixed-v0",  "elbow_standard", {"r_elbow_flex": (2.0, 2.0)}, "init")
_register_pose_env("myoElbowPose1D6MRandom-v0", "elbow_standard", {"r_elbow_flex": (0.0, 2.27)})
```

Registration file shrinks from ~780 lines to ~200. New variants require one new line.

---

## Finding 5 — Hardcoded Joint/Site/Anatomy Names

**Severity: Medium | Effort: M | Files: 5+**

Anatomical strings are scattered across `mjx/__init__.py`, `walk_env.py`, and
`integrations/musclemimic/bimanual_model.py`. Typos fail silently at model load
time (MuJoCo throws a cryptic error).

```python
# mjx/__init__.py:296 — 23 hand joints hardcoded inline
cfg["target_jnt_range"] = config_dict.create(
    pro_sup=jp.array((-0.3, 0.7)),
    deviation=jp.array((-0.2, 0.3)),
    flexion=jp.array((-0.8, 0.2)),
    cmc_abduction=jp.array((-0.2, 0.8)),
    # ... 19 more
)

# walk_env.py:110 — joint lookups hardcoded
self._hip_flex_l_adr = _qadr("hip_flexion_l")
self._hip_flex_r_adr = _qadr("hip_flexion_r")

# bimanual_model.py:28 — body→site mapping hardcoded dict
BODY2SITES_FOR_MIMIC = {
    "thorax": "upper_body_mimic",
    "humerus_l": "left_shoulder_mimic",
    ...  # 10 entries
}
```

**Recommended pattern:** A per-model YAML or `model_anatomy.py` metadata module:

```python
# myosuite/models/anatomy/hand_anatomy.py
HAND_JOINT_ROM: dict[str, tuple[float, float]] = {
    "pro_sup":       (-0.3, 0.7),
    "deviation":     (-0.2, 0.3),
    "flexion":       (-0.8, 0.2),
    "cmc_abduction": (-0.2, 0.8),
    ...
}
```

This also serves as documentation of the physiological range-of-motion values
and can be validated programmatically against the loaded MjModel.

---

## Finding 6 — String-Keyed Term Lookup with Convention Magic

**Severity: Medium | Effort: S | Files: 2**

`modular_env.py` resolves term names by appending a suffix and using `getattr()`:

```python
# modular_env.py:56
fn_name = f"{key}_obs"            # "joint_pos" → looks for "joint_pos_obs"
fn = getattr(myo_obs_terms, fn_name)

# modular_env.py:90
for fn_name in (f"{term}_reward", term):  # "pose" → tries "pose_reward" then "pose"
    if hasattr(myo_reward_terms, fn_name):
        return getattr(myo_reward_terms, fn_name)
```

This convention works, but breaks silently when term names collide or the suffix
rule changes. `mjx_modular_env.py:51` duplicates the same logic.

**Recommended pattern:** Build registries at import time via reflection — done once,
used everywhere, no duplicate lookup logic:

```python
# myosuite/terms/__init__.py
import myosuite.terms.myo_obs_terms as _obs
import myosuite.terms.myo_reward_terms as _rwd

OBS_TERMS: dict[str, Callable] = {
    name.removesuffix("_obs"): fn
    for name, fn in vars(_obs).items()
    if name.endswith("_obs") and callable(fn)
}
REWARD_TERMS: dict[str, Callable] = {
    name.removesuffix("_reward"): fn
    for name, fn in vars(_rwd).items()
    if name.endswith("_reward") and callable(fn)
}
```

Then `modular_env.py` does `fn = OBS_TERMS[key]` — one line, fails fast with
a clear `KeyError`, no convention knowledge required. Duplication in
`mjx_modular_env.py` disappears.

---

## Finding 7 — Hardcoded Reward Weights and Physics Thresholds

**Severity: Medium | Effort: S | Files: 5+**

Numeric constants appear as anonymous literals in both config dicts and term
function bodies:

```python
# mjx/__init__.py:71 — pose reward config
reward_config=config_dict.create(
    angle_reward_weight=1.0,  ctrl_cost_weight=1.0,
    pose_thd=0.35,            far_th=4 * jp.pi / 2,
    bonus_weight=4.0,
)

# myo_reward_terms.py:45 — bonus thresholds
bonus = 1.0 * (dist < pose_thd) + 1.0 * (dist < 1.5 * pose_thd)
penalty = -1.0 * (dist > 2 * math.pi)

# mjx/__init__.py:311,320 — per-env far_th overrides
cfg["far_th"] = 0.044   # HandReachFixed
cfg["far_th"] = 0.034   # HandReachRandom
```

The `1.5` multiplier in `myo_reward_terms.py` is undocumented — it is unclear
whether it was tuned or arbitrary.

**Recommended pattern:** Named constants with inline rationale comments:

```python
# myosuite/terms/reward_constants.py
POSE_BONUS_NEAR_MULT   = 1.0   # per-unit bonus at dist < pose_thd
POSE_BONUS_FAR_MULT    = 1.5   # wider acceptance band for partial credit
POSE_PENALTY_THRESHOLD = 2     # penalty multiplier on pi (full rotation)
```

This is a low-friction win: no code structure changes, just named constants
replacing literals.

---

## Finding 8 — Env ID Naming Conventions Are String-Sliced

**Severity: Medium | Effort: S | Files: 2**

Variant ID generation relies on `id[:3]` string slices baked into two separate
registration helpers:

```python
# myobase/__init__.py:30
if id[:3] == "myo":
    variant_id = id[:3] + "Sarc" + id[3:]   # "myoElbow..." → "myoSarcElbow..."

# core/registry.py:196
if base_env_id.startswith("myo"):
    variant_id = "myo" + vspec.suffix + base_env_id[3:]
```

These two implementations are not in sync — `myobase` uses prefix+Sarc+rest while
`registry.py` uses prefix+suffix+rest. Changing the convention means hunting both.

**Recommended pattern:** One utility function owned by the registry:

```python
# core/registry.py
def _variant_id(base_id: str, suffix: str) -> str:
    """Insert suffix after the 'myo'/'motor'/'Mjx' prefix."""
    for prefix in ("myo", "motor", "Mjx"):
        if base_id.startswith(prefix):
            return f"{prefix}{suffix}{base_id[len(prefix):]}"
    base, ver = base_id.rsplit("-", 1)
    return f"{base}{suffix}-{ver}"
```

Both callers delegate here. Convention change → one line edit.

---

## Finding 9 — Backend Selection Logic Scattered

**Severity: Low-Medium | Effort: S | Files: 2**

`core/registry.py:make_env()` dispatches backends via `if backend == "cpu"` /
`elif backend == "mjx"` / `elif backend == "mjlab"` with conditional imports
inside each branch. `register_task()` has a separate `if "mjx" in backends:`
check. The backend list is not enumerated anywhere as a canonical constant.

```python
# registry.py:237
if backend == "cpu":
    return gym.make(env_id, **overrides)
elif backend == "mjx":
    try:
        from mujoco_playground import registry as pg_registry
        return pg_registry.load(env_id)
    except ImportError ...
elif backend == "mjlab": ...
else:
    raise ValueError(f"Unknown backend: {backend!r}. Choose from: cpu, mjx, mjlab")
```

The valid backend names (`"cpu"`, `"mjx"`, `"mjlab"`) live only in the error string.

**Recommended pattern:** Declare canonical backend names as a constant; lazy-import
via a dispatch table:

```python
SUPPORTED_BACKENDS = frozenset({"cpu", "mjx", "mjlab"})

_BACKEND_MAKERS: dict[str, Callable] = {}  # populated by each backend module on import

def make_env(env_id, backend="cpu", **overrides):
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}. Choose from: {SUPPORTED_BACKENDS}")
    return _BACKEND_MAKERS[backend](env_id, **overrides)
```

---

## Finding 10 — Test Expectations Hardcoded Against Registry

**Severity: Low | Effort: S | Files: 1**

`test_entry_points.py` tests MJX env completeness by comparing `ALL_ENVS` against a
hardcoded expected set, meaning the test must be manually updated when envs are added:

```python
# test_entry_points.py:133
expected = {
    "MjxElbowPoseFixed-v0",
    "MjxElbowPoseRandom-v0",
    "MjxFingerPoseFixed-v0",
    "MjxFingerPoseRandom-v0",
    "MjxHandReachFixed-v0",
    "MjxHandReachRandom-v0",
}
missing = expected - set(ALL_ENVS)
```

The check is redundant — `ALL_ENVS` is already the source of truth.

**Recommended pattern:** Remove the hardcoded set; test structural properties instead:

```python
assert len(ALL_ENVS) >= 6, "Expected at least 6 MJX environments"
assert all(e.startswith("Mjx") and e.endswith("-v0") for e in ALL_ENVS)
```

---

## Finding 11 — Observation Dimension Comments Not Validated

**Severity: Low | Effort: S | Files: 1**

`walk_env.py` docstring enumerates observation components with hardcoded dimension
counts (nq=35, nv=34, nu=80, total=403). These are not asserted anywhere; a model
change would silently produce wrong-shaped observations while the comment stays stale.

**Recommended pattern:** Add a single assertion at `setup_model()` time:

```python
def _expected_obs_dim(model) -> int:
    return (model.nq - 2) + model.nv + 2 + 4 + 2 + 1 + 6 + 1 + 3 * model.nu + model.na

# in setup_model():
assert obs_vec.shape[-1] == _expected_obs_dim(mj_model), (
    f"Walk obs shape mismatch: got {obs_vec.shape[-1]}, "
    f"expected {_expected_obs_dim(mj_model)}"
)
```

---

## Finding 12 — Anatomy Metadata Not Co-Located with Models

**Severity: Low | Effort: M | Files: 5+**

Body→site mappings, joint groups, and muscle token lists are defined as module-level
dicts in `integrations/musclemimic/bimanual_model.py` (BODY2SITES_FOR_MIMIC,
FINGER_JOINT_TOKENS, FINGER_MUSCLE_TOKENS). These are logically properties of the
MJCF model files, not Python source.

If the XML changes (e.g. a site is renamed), the Python constants go stale with no
validation. The same names must be duplicated if the fullbody variant adds new joints.

**Recommended pattern:** Co-locate metadata as YAML sidecar files next to each XML:

```
simhive/myo_sim/hand/myohand_pose.xml
simhive/myo_sim/hand/myohand_pose.metadata.yaml  ← joint groups, site mappings
```

Load at model instantiation:
```python
metadata = yaml.safe_load(model_path.with_suffix(".metadata.yaml").read_text())
JOINT_GROUPS = metadata["joint_groups"]
```

This makes anatomy data visible alongside the model, version-controlled together,
and validatable at load time (check that every YAML key exists in the model).

---

## Prioritized Action Plan

### Tier 1 — Small effort, large blast-radius reduction (do this sprint)

| # | Finding | Key change | Est. |
|---|---------|-----------|------|
| 1 | **§2 Asset paths** | `MODEL_ASSET_MAP` dict in `model_recipes.py` | 2h |
| 2 | **§3 MJX registration** | Replace `make()` if/elif with `_MJX_ENV_SPECS` table | 3h |
| 3 | **§6 Term registry** | Auto-build `OBS_TERMS`/`REWARD_TERMS` from module reflection in `terms/__init__.py` | 1h |
| 4 | **§4 Registration factories** | `_register_pose_env()` / `_register_reach_env()` in `myobase/__init__.py` | 2h |

### Tier 2 — Medium effort, significant maintainability wins (next sprint)

| # | Finding | Key change | Est. |
|---|---------|-----------|------|
| 5 | **§8 Env ID naming** | Single `_variant_id()` helper in `registry.py` | 1h |
| 6 | **§7 Reward constants** | `reward_constants.py` with named physics values | 2h |
| 7 | **§9 Backend dispatch** | `SUPPORTED_BACKENDS` frozenset + dispatch table stub | 2h |
| 8 | **§10 Test expectations** | Remove hardcoded expected set in test_entry_points.py | 30m |

### Tier 3 — Larger refactors (plan for v1.0 cleanup)

| # | Finding | Key change | Est. |
|---|---------|-----------|------|
| 9 | **§1 Hardware dispatch** | `HardwareInterface` ABC + `HARDWARE_DRIVERS` registry | 1 day |
| 10 | **§5 Joint metadata** | Per-model YAML anatomy sidecars | 2 days |
| 11 | **§11 Walk obs validation** | `_expected_obs_dim()` assertion | 1h |
| 12 | **§12 Anatomy co-location** | YAML metadata adjacent to XML models | 1 day |

---

## Key Takeaway

The codebase has good modularity *within* the new architecture (TaskConfig, term
functions, EnvAccessor). The hardcoding debt is concentrated in the *edges*:
hardware interface layer (robot.py), env registration glue (myobase/__init__.py,
mjx/__init__.py), and anatomy metadata. Tier 1 items can be addressed without
touching any env logic — they are pure data extraction from existing if/elif chains
into dicts or factory functions.
