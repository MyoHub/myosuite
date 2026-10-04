# Writing Term Functions

Term functions are pure functions that read physics state through an `EnvAccessor` and run identically on CPU (numpy), MJX (jax.numpy), and mjlab (torch).

All term functions live in `myosuite/terms/`.

## Signatures

| Kind | Signature | Return must include |
|---|---|---|
| Obs | `(accessor, **kwargs) -> dict[str, Any]` | any named arrays |
| Reward | `(accessor, obs_dict, **kwargs) -> dict` | `dense`, `solved`, `done` |
| Termination | `(accessor, obs_dict, **kwargs) -> dict` | `done` |
| Action | `(accessor, action, **kwargs) -> Any` | processed action |

## The Only Rule: `accessor.array_module()`

Never import `numpy`, `jax.numpy`, or `torch` directly. Get the right array library at runtime:

```python
def distance_reward(accessor, obs_dict, *, threshold: float = 0.05, **kwargs):
    xp = accessor.array_module()
    dist = float(xp.linalg.norm(obs_dict["site_err"]))
    return {"dense": -dist, "solved": dist < threshold, "done": False}
```

Access physics state via `accessor.*` only — never `mujoco.MjData` directly:

```python
qpos  = accessor.joint_pos()          # (nq,) or (N, nq)
qvel  = accessor.joint_vel()          # (nv,) or (N, nv)
sites = accessor.site_xpos(site_ids)  # (K, 3)
t     = accessor.time()
qids, ranges = accessor.joint_range() # limited hinge/slide joints: qpos indices, (k, 2) ranges
force = accessor.muscle_force()       # MuJoCo muscles only (tension < 0); also muscle_length/velocity
params = accessor.muscle_params()     # static MuscleParams (F0, L0, curve parameters) of those muscles
tau   = accessor.qfrc_actuator()      # actuator forces in joint space, joint_vel() layout
```

Full protocol: `myosuite/core/protocols.py`. Joint limits come from `joint_range()`, never from
`ctrl_range()` (actuator controls; nq != nu in general).

## Effort and ergonomics terms

`myosuite/terms/effort.py` (opt-in; mjlab reward wrappers in `backends/mjlab/tasks/mdp/effort.py`):

| Term | Measure | Reference |
|---|---|---|
| `muscle_mechanical_power` | sum of `abs(F v)` (or positive work only) over the muscles | Berret et al. 2011, PLoS Comput Biol 7:e1002183; Margaria 1968 |
| `metabolic_energy_rate` | muscle heat + work rate (W), `version="2003"` / `"2010"` | Umberger, Gerritsen & Martin 2003, CMBBE 6:99; Umberger 2010, J R Soc Interface 7:1329; OpenSim `Umberger2010MuscleMetabolicsProbe` (Uchida et al. 2016) |
| `consumed_endurance`, `endurance_time`, `consumed_endurance_episode` | shoulder torque / Max_Torque, endurance time, CE (%) | Hincapié-Ramos et al. 2014, CHI, Eqs. 1-2 (endurance), 7 (CE); Max_Torque 22.94 / 18.57 N m as reported by Li et al. 2024, TOCHI |
| `fatigue_effort` | mean / max fatigued fraction MF, `norm(MA - TL)` | Xia & Frey-Law 2008, J Biomech 41:3046; Looft et al. 2018, J Biomech 77:16 |
| `joint_limit_discomfort` | smooth squared hinge in the outer `margin` of each joint range | generic form (cf. Marler et al. 2005, SAE 2005-01-2680) |

MuJoCo assumptions of the metabolic model: rigid tendon (fiber velocity = actuator velocity),
MuJoCo's force-length curves, muscle mass `F0 / 0.25 MPa * 1059.7 kg/m^3 * L0`, 50 % fast-twitch
fibres, `vmax_FT` = the muscle's `vmax`, excitation = activation unless `task_state["muscle_excitation"]`.

## Minimal Example

```python
def pose_tracking_reward(
    accessor, obs_dict, *, target_key="target_qpos", weight=1.0, threshold=0.05, **kwargs
) -> dict:
    xp = accessor.array_module()
    dist = float(xp.linalg.norm(obs_dict[target_key] - accessor.joint_pos()))
    return {"pose_err": dist, "dense": -weight * dist, "solved": dist < threshold, "done": False}
```

Register in a `TaskSpec`:
```python
TaskSpec(reward_terms=[(pose_tracking_reward, {"weight": 2.0})])
```

## Testing

```python
# myosuite/tests/test_terms_cpu.py
def test_pose_tracking_reward(mock_cpu_accessor, mock_obs_dict):
    result = pose_tracking_reward(mock_cpu_accessor, mock_obs_dict)
    assert {"dense", "solved", "done"} <= result.keys()
    assert isinstance(result["dense"], float)
```

Add a CPU parity test in `test_parity.py` (default `atol=1e-6`). For a GPU
task, also honour `cross-backend-contract.md` so the mjlab half stays in lockstep.
