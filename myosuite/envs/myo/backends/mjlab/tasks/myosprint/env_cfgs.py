# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Task config for myoSprint-v0: Bolt's full-body sprinter model on the mjlab/MuJoCo Warp backend, driven
by myo_sim.warp's Bolt-parity muscle/contact physics (not myosuite's own muscle model -- see
../actuators/bolt_muscle_action.py for why).

This is a first, minimal working env config (Phase 4 of the myo_bolt_vv porting plan is still in progress):
observations are raw qpos/qvel rather than the full 336-D Bolt-convention observation. Sufficient to
construct a working ManagerBasedRlEnv end to end and gate BoltMuscleAction's physics against myo_sim.warp
directly; the full Bolt-parity observation/transfer validation are tracked separately (Phases 4-6).

Control timing: decimation=333 at sim timestep=1e-4 gives Bolt's own ~30 Hz control rate
(scripts/msk_envs_repro/cpu_td3_side_by_side.py's CTRL_DT) -- required for a Bolt-trained policy (tuned at
that rate) to see the timing it expects; an earlier placeholder of decimation=10 here was 33x too fine.

Reward lambdas cross-checked against a sibling exploration (myosuite branch
feature/myofullbody-simple-sprint, athletic_contract.py's AthleticSpec -- itself sourced from Bolt/
msk_envs_repro's own SprintConfig/LaneConfig): lambda_vel=1e-2, lambda_limit=-3e-4. Two things worth noting
from that cross-check: (1) lambda_limit=-3e-4 matches this session's own earlier finding that a GPU reward
bug had it set to +1 instead; (2) that branch's comment states "Bolt has no explicit alive term" -- taken at
face value here (alive_bonus=0), overriding this file's earlier alive-bonus placeholder.
"""
from __future__ import annotations

import torch
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.entity import EntityCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim.sim import SimulationCfg, MujocoCfg
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.envs.mdp.terminations import time_out

from myosuite.envs.myo.backends.mjlab.actuators.bolt_muscle_action import BoltMuscleActionCfg

# Bolt / msk_envs SprintConfig lambdas (cross-checked against athletic_contract.py, see module docstring).
LAMBDA_VEL = 1e-2
LAMBDA_LIMIT = -3e-4
SIM_DT = 1e-4
CTRL_DT = 1.0 / 30.0  # Bolt sprint's own ~30 Hz control rate (scripts/msk_envs_repro/cpu_td3_side_by_side.py)
DECIMATION = round(CTRL_DT / SIM_DT)  # 333 -- NOT an arbitrary choice: this is what makes a Bolt-trained
# policy (whose actions were tuned at 30 Hz) see the timing it expects. An earlier placeholder of 10 here
# would have run the policy 33x faster than the control rate it was trained at.
FALL_HEIGHT = 0.55  # pelvis y (this model's up axis) below which an episode ends -- matches the fixed-body
# fall threshold used throughout myo_bolt_vv's own sprint trainers (scripts/msk_envs_repro/warp_sprint_env.py).
SPRINT_ENTITY = "sprinter"


def _sprint_spec_fn():
    import myo_sim

    return myo_sim.load_spec("myosprinter")


# Bone mesh set for a real skeletal render (myosprinter's own spec carries no visual geoms at all -- it's
# collision/physics only, since this suite's own ContactSystem does collision from its own geom_flat/pairs
# table, not MuJoCo's native contype/conaffinity). Meshes come *only* from myo_sim's own bundled asset
# library (myo_sim/models/meshes/*.stl -- the same generic-OpenSim-skeleton STL set myo_sim's own leg/arm
# models attach at identity pos/quat/scale, e.g. myo_sim/models/leg/assets/myolegs_chain.xml's
# `<geom mesh="r_femur" .../>` inside the femur_r body -- confirming these meshes are already authored in
# each body's own local frame, the same convention Bolt's OpenSim-derived sprinter bodies use). No files
# from outside myo_sim, and nothing copied in.
_MESH_DIR = None  # resolved lazily from myo_sim.MODELS_DIR / "meshes" (see _mesh_dir())

# myosprinter's body name -> myo_sim mesh file(s). Bolt's sprinter has no separate head/neck/spine bodies
# (see full_bolt.osim's own <Body> list: 23 bodies, no head/neck/spine -- "torso" is the whole
# Head-Arms-Trunk segment, the standard OpenSim gait-model simplification), so the torso entry below pulls
# in myo_sim's own HAT mesh set (hat_skull/hat_cervical/hat_thoracic/hat_lumbar/hat_ribs_scap) -- visual
# only, no new physics body, same as any other body's own extra mesh piece.
_MESH_MAP = {
    "pelvis": ["r_pelvis.stl", "l_pelvis.stl"],
    "femur_r": ["r_femur.stl"],
    "femur_l": ["l_femur.stl"],
    "tibia_r": ["r_tibia.stl", "r_fibula.stl"],
    "tibia_l": ["l_tibia.stl", "l_fibula.stl"],
    "talus_r": ["r_talus.stl"],
    "talus_l": ["l_talus.stl"],
    "calcn_r": ["r_foot.stl"],
    "calcn_l": ["l_foot.stl"],
    "toes_r": ["r_bofoot.stl"],
    "toes_l": ["l_bofoot.stl"],
    # torso_lowpoly-v1.stl and the hat_*.stl set turned out to be authored in an absolute ground/display
    # frame (verified directly: torso_lowpoly-v1.stl's own bounding box is y=[0.69, 1.67], matching a
    # standing human's hip-to-shoulder height in world coordinates, not a small body-local offset -- myo_sim
    # itself only uses these as worldbody-attached reference geoms, e.g. ground_spine.stl/ground_skull.stl).
    # thorax.stl is myo_sim's actual body-local ribcage mesh (myotorso_chain.xml:524, pos="0.028 0.365 0"
    # relative to its parent body -- the only real calibration data point available), used here as the best
    # available approximation for Bolt's single rigid torso body. No calibrated body-local offset was found
    # for a head/neck/spine piece specifically (the hat_* set is ground-frame only), so head/neck/spine is
    # dropped rather than guessed at and risk another disconnected-looking render.
    "torso": [("thorax.stl", (0.028, 0.365, 0.0))],
    "clavicle_r": ["clavicle.stl"],
    "clavicle_l": ["clavicle.stl"],  # myo_sim ships one clavicle/scapula/humerus/ulna/radius mesh, not
    "scapula_r": ["scapula.stl"],  # separate left/right pairs (unlike the leg meshes, which are genuinely
    "scapula_l": ["scapula.stl"],  # l_/r_ prefixed) -- myo_sim's own build mirrors the whole arm spec at
    "humerus_r": ["humerus.stl"],  # the body level (build_mirrored_left_arm_spec in compose.py) rather than
    "humerus_l": ["humerus.stl"],  # per-mesh; reusing the same unmirrored mesh on both sides here is an
    "ulna_r": ["ulna.stl"],  # accepted simplification -- the left arm won't be mirrored, just reused as-is.
    "ulna_l": ["ulna.stl"],
    "radius_r": ["radius.stl"],
    "radius_l": ["radius.stl"],
    # hand_r/hand_l: no hand mesh in myo_sim/models/meshes -- left unmeshed rather than approximating with
    # something anatomically wrong.
}


def _mesh_dir() -> str:
    import myo_sim

    return str(myo_sim.MODELS_DIR / "meshes")


def _attach_bone_meshes(spec, verbose: bool = True):
    """Mutates `spec` in place, adding visual-only bone mesh geoms (contype=0, conaffinity=0 -- this
    suite's ContactSystem does collision from its own table, so these have no physics effect), using only
    myo_sim's own bundled meshes (see _MESH_MAP above)."""
    import mujoco
    from pathlib import Path

    mesh_dir = _mesh_dir()
    spec.meshdir = mesh_dir
    bodies = {b.name: b for b in spec.bodies if b.name}
    used = 0
    for bname, files in _MESH_MAP.items():
        body = bodies.get(bname)
        if body is None:
            continue
        for k, entry in enumerate(files):
            fname, pos = entry if isinstance(entry, tuple) else (entry, (0.0, 0.0, 0.0))
            if not (Path(mesh_dir) / fname).is_file():
                raise FileNotFoundError(f"{fname} not found in myo_sim's own {mesh_dir}")
            mname = f"{bname}__{k}_{Path(fname).stem}"
            mesh = spec.add_mesh()
            mesh.name = mname
            mesh.file = fname
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mname,
                pos=list(pos),
                rgba=[0.88, 0.84, 0.74, 1.0],
                contype=0,
                conaffinity=0,
                group=0,
            )
            used += 1
    if verbose:
        print(f"attached {used} bone meshes (myo_sim's own meshes only) from {mesh_dir}")
    return spec


def _add_floor_and_lights(spec):
    """mjlab's OffscreenRenderer doesn't add a floor or lights by default -- add both directly to the
    visual spec so the render isn't a skeleton floating in a black void."""
    import mujoco

    # Sprint covers ~18m in x; a fixed-position worldbody light stays put while the runner moves far away
    # from it, so lighting fades over the clip. Headlight (camera-relative, always-on ambient/diffuse/
    # specular) doesn't have that problem -- boost it well above MuJoCo's dim defaults instead of relying
    # on worldbody point lights for the primary illumination.
    spec.visual.headlight.ambient = [0.5, 0.5, 0.5]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.specular = [0.2, 0.2, 0.2]

    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[40, 40, 0.1],
        # This model is Y-up (gravity along -y); the plane's default normal is local +Z, so rotate -90 deg
        # about X to point it along world +Y (same convention used throughout myo_bolt_vv's own renders).
        quat=[0.70710678, -0.70710678, 0, 0],
        rgba=[0.55, 0.55, 0.58, 1],
        contype=0,
        conaffinity=0,
    )
    return spec


def _sprint_spec_fn_visual():
    """Same physics spec as _sprint_spec_fn(), plus real bone meshes and a floor/lights for rendering."""
    return _add_floor_and_lights(_attach_bone_meshes(_sprint_spec_fn()))


def _init_state() -> EntityCfg.InitialStateCfg:
    """Standing pose, not mjlab's zero default.

    mjlab's own InitialStateCfg default (pos=(0,0,0)) silently overrides the compiled spec's own qpos0 on
    every reset -- without this, the pelvis starts co-located with the ground contact geoms and immediately
    blows up (NaN within the first step). Tried reading the pose from the model's own qpos0/keyframe first
    (mirroring the feature/myofullbody-simple-sprint cross-check's ``_init_state()``, which works for its
    model), but full_free_noplugin.xml carries no keyframe and a trivial all-zero qpos0 -- myo_bolt_vv's own
    scripts always set the 1.05 m standing pelvis-ty height programmatically at runtime, never baked into
    the XML (scripts/msk_envs_repro/bolt_sprint_parity.py and friends), so that pattern doesn't transfer
    here. Hardcoding the same runtime value instead.
    """
    return EntityCfg.InitialStateCfg(pos=(0.0, 1.05, 0.0))


def _obs_qpos(env) -> torch.Tensor:
    return torch.as_tensor(env.sim.wp_data.qpos.numpy(), device=env.device)


def _obs_qvel(env) -> torch.Tensor:
    return torch.as_tensor(env.sim.wp_data.qvel.numpy(), device=env.device)


def _pelvis_y(env) -> torch.Tensor:
    # qpos[:, 1] is the pelvis y coordinate (this model's up axis) -- freejoint pelvis is qpos[0:3].
    return torch.as_tensor(env.sim.wp_data.qpos.numpy()[:, 1], device=env.device)


def _forward_vel_reward(env) -> torch.Tensor:
    # qvel[:, 0] is the pelvis x (forward) linear velocity.
    return torch.as_tensor(env.sim.wp_data.qvel.numpy()[:, 0], device=env.device)


def _limit_pain(env) -> torch.Tensor:
    """Soft hinge-limit overshoot sum (positive when a joint is past its range); weighted by LAMBDA_LIMIT<0,
    i.e. it's a penalty. Bolt's own r_pain term -- see the paper's r = r_speed - w*r_pain + r_alive."""
    import mujoco

    sim = env.sim
    m = sim.mj_model
    qpos = sim.wp_data.qpos.numpy()
    pain = torch.zeros(qpos.shape[0], device=env.device)
    for j in range(int(m.njnt)):
        jtype = int(m.jnt_type[j])
        if jtype not in (int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE)):
            continue
        lo, hi = float(m.jnt_range[j, 0]), float(m.jnt_range[j, 1])
        if not (hi > lo) or abs(hi - lo) > 1e6:
            continue
        adr = int(m.jnt_qposadr[j])
        q = torch.as_tensor(qpos[:, adr], device=env.device)
        pain = pain + torch.clamp(torch.tensor(lo, device=env.device) - q, min=0.0)
        pain = pain + torch.clamp(q - torch.tensor(hi, device=env.device), min=0.0)
    return pain


def _fall_termination(env) -> torch.Tensor:
    return _pelvis_y(env) < FALL_HEIGHT


def make_sprint_env_cfg(
    num_envs: int = 64,
    init_state: EntityCfg.InitialStateCfg | None = None,
    default_activation: float | None = None,
    visual: bool = False,
) -> ManagerBasedRlEnvCfg:
    """A minimal but real ManagerBasedRlEnvCfg for the Bolt-parity sprinter, driven by BoltMuscleAction.

    init_state: defaults to the neutral standing pose (_init_state()). Pass an explicit InitialStateCfg to
    override it -- e.g. Phase 6's transfer-validation test needs Bolt's own "running start" pose
    (scripts/msk_envs_repro/starting_pose_run.yaml: nonzero forward/hip/knee velocity, not neutral), which
    the TD3 checkpoint was trained to begin from; a neutral zero-velocity standing start is a different
    initial condition the policy was never trained on and falls within a few steps from it (observed
    directly: xT=-0.16m, fell at step 11, moving backward -- not a physics bug, a mismatched starting state).

    visual: False (default, used for training/testing) compiles the plain physics-only spec (myosprinter
    has no visual geoms at all -- faster to compile/step at scale). True attaches real bone meshes
    (_sprint_spec_fn_visual(), including head/neck/spine on the torso body) for rendering -- see
    render_myosprint_mjlab.py.
    """
    spec_fn = _sprint_spec_fn_visual if visual else _sprint_spec_fn
    entity_cfg = EntityCfg(spec_fn=spec_fn, init_state=init_state or _init_state())
    scene_cfg = SceneCfg(num_envs=num_envs, env_spacing=3.0, entities={SPRINT_ENTITY: entity_cfg})

    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "qpos": ObservationTermCfg(func=_obs_qpos),
                "qvel": ObservationTermCfg(func=_obs_qvel),
            },
        ),
    }
    actions = {
        "bolt_muscles": BoltMuscleActionCfg(entity_name=SPRINT_ENTITY, default_activation=default_activation),
    }
    rewards = {
        "forward_vel": RewardTermCfg(func=_forward_vel_reward, weight=LAMBDA_VEL),
        "limit_pain": RewardTermCfg(func=_limit_pain, weight=LAMBDA_LIMIT),
    }
    terminations = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
        "fall": TerminationTermCfg(func=_fall_termination, time_out=False),
    }
    events = {
        "reset_scene_to_default": EventTermCfg(func=reset_scene_to_default, mode="reset"),
    }

    return ManagerBasedRlEnvCfg(
        decimation=DECIMATION,
        scene=scene_cfg,
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        events=events,
        episode_length_s=10.0,
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=SIM_DT,
                # This model is Y-up (gravity along -y, matching Bolt's own convention), not mjlab's Z-up
                # default -- SimulationCfg.mujoco.apply() unconditionally overwrites model.opt.gravity, so
                # it must be set explicitly here or the model falls sideways relative to its own up axis.
                # (The feature/myofullbody-simple-sprint cross-check doesn't need this override -- myo_sim's
                # own composed models are Z-up; this Y-up quirk is specific to Bolt's own OpenSim asset.)
                gravity=(0.0, -9.80665, 0.0),
                ccd_iterations=500,
            ),
            # Generous headroom for this contact-rich model (27 geoms, 297 pairs) -- avoids
            # "solver iterations limit reached" warnings seen during Phase 0 testing at the defaults.
            njmax=768,
            nconmax=384,
        ),
    )
