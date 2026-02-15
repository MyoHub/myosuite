"""
Diagnostic script for MyoSkeleton model tuning validation.

Runs a series of checks to determine whether myoskeleton_edited.xml is
properly configured for RL training (e.g. with PPO via SB3).

Usage:
    python -m myosuite.envs.myo.myoskeleton.diagnose_model

Checks performed:
    1. Model loads and env resets without error
    2. Simulation stability (passive drop test + zero-action rollout)
    3. Actuator authority (can torques meaningfully move joints?)
    4. Observation / action space sanity
    5. Reward signal quality (non-degenerate, varies with state)
    6. Gravity compensation feasibility
    7. Reference motion playback fidelity
"""

import sys
import numpy as np
import mujoco
import gymnasium as gym

# Ensure myosuite envs are registered
import myosuite  # noqa: F401


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _ok(msg: str):
    print(f"  [PASS] {msg}")


def _warn(msg: str):
    print(f"  [WARN] {msg}")


def _fail(msg: str):
    print(f"  [FAIL] {msg}")


# ---------------------------------------------------------------------------
# 1. Basic loading
# ---------------------------------------------------------------------------
def check_loading(env_name: str):
    _header("1. Environment Loading")
    try:
        env = gym.make(env_name)
        _ok(f"gym.make('{env_name}') succeeded")
    except Exception as e:
        _fail(f"gym.make failed: {e}")
        return None

    try:
        obs, info = env.reset()
        _ok(f"env.reset() succeeded, obs shape = {obs.shape}")
    except Exception as e:
        _fail(f"env.reset() failed: {e}")
        return None

    return env


# ---------------------------------------------------------------------------
# 2. Simulation stability
# ---------------------------------------------------------------------------
def check_stability(env):
    _header("2. Simulation Stability (zero-action rollout)")
    env.reset()
    model = env.unwrapped.mj_model
    data = env.unwrapped.mj_data

    n_steps = 500
    zero_action = np.zeros(env.action_space.shape)
    rewards = []
    z_positions = []
    qpos_norms = []
    qvel_norms = []
    nan_detected = False

    for step in range(n_steps):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        rewards.append(reward)
        z_positions.append(data.qpos[2])
        qpos_norms.append(np.linalg.norm(data.qpos))
        qvel_norms.append(np.linalg.norm(data.qvel))

        if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
            _fail(f"NaN detected at step {step}")
            nan_detected = True
            break
        if terminated:
            _warn(f"Episode terminated at step {step} (z={data.qpos[2]:.3f})")
            break

    if not nan_detected:
        _ok("No NaN values during zero-action rollout")

    z_arr = np.array(z_positions)
    qvel_arr = np.array(qvel_norms)
    print(f"  Root Z: start={z_arr[0]:.3f}, end={z_arr[-1]:.3f}, "
          f"min={z_arr.min():.3f}, max={z_arr.max():.3f}")
    print(f"  Qvel norm: mean={qvel_arr.mean():.3f}, max={qvel_arr.max():.3f}")

    # Check if model collapses immediately under gravity
    if z_arr[-1] < 0.3:
        _fail("Model collapsed to ground — joints may lack sufficient "
              "stiffness/damping or initial pose is unstable")
    elif z_arr[-1] < 0.6:
        _warn("Model sank significantly — may need higher joint stiffness "
              "or damping to hold standing pose passively")
    else:
        _ok("Model maintains reasonable height under zero action")

    # Check for exploding velocities
    if qvel_arr.max() > 100:
        _fail(f"Velocities exploded (max norm={qvel_arr.max():.1f}). "
              "Reduce timestep, increase damping, or check actuator gains.")
    elif qvel_arr.max() > 50:
        _warn(f"High velocities detected (max norm={qvel_arr.max():.1f}). "
              "Consider increasing damping.")
    else:
        _ok(f"Velocities remain bounded (max norm={qvel_arr.max():.1f})")

    return rewards, z_positions


# ---------------------------------------------------------------------------
# 3. Actuator authority
# ---------------------------------------------------------------------------
def check_actuator_authority(env):
    _header("3. Actuator Authority")
    model = env.unwrapped.mj_model
    data = env.unwrapped.mj_data

    n_actuators = model.nu
    print(f"  Number of actuators: {n_actuators}")
    print(f"  Number of joints: {model.njnt}")
    print(f"  Degrees of freedom (nv): {model.nv}")
    print(f"  Generalized positions (nq): {model.nq}")

    # Ratio of actuated to total DOF
    actuator_ratio = n_actuators / model.nv
    if actuator_ratio < 0.2:
        _warn(f"Only {actuator_ratio:.1%} of DOFs are actuated "
              f"({n_actuators}/{model.nv}). Many joints are passive — "
              "the policy can only control a subset of the body.")
    else:
        _ok(f"Actuator coverage: {actuator_ratio:.1%} of DOFs")

    # Test each actuator individually
    weak_actuators = []
    strong_actuators = []
    for i in range(n_actuators):
        env.reset()
        initial_qpos = data.qpos.copy()

        # Apply max positive action to this actuator only
        action = np.zeros(n_actuators)
        action[i] = 1.0  # max positive

        total_delta = np.zeros(model.nq)
        for _ in range(50):  # 50 steps ≈ 1 second at frame_skip=10
            env.step(action)
            if np.any(np.isnan(data.qpos)):
                break

        delta = np.linalg.norm(data.qpos - initial_qpos)
        act_name = model.actuator(i).name

        if delta < 0.001:
            weak_actuators.append((act_name, delta))
        elif delta > 10.0:
            strong_actuators.append((act_name, delta))

    if weak_actuators:
        _warn(f"{len(weak_actuators)} actuators produce negligible motion:")
        for name, d in weak_actuators[:5]:
            print(f"    - {name}: delta={d:.6f}")
        if len(weak_actuators) > 5:
            print(f"    ... and {len(weak_actuators) - 5} more")
    else:
        _ok("All actuators produce measurable motion")

    if strong_actuators:
        _warn(f"{len(strong_actuators)} actuators produce excessive motion "
              "(delta > 10):")
        for name, d in strong_actuators[:5]:
            print(f"    - {name}: delta={d:.4f}")
    else:
        _ok("No actuators produce excessively large motion")

    # Print actuator summary table
    print("\n  Actuator summary:")
    print(f"  {'Name':<40} {'Gear':>6} {'ForceRange':>16} {'GainPrm[0]':>10}")
    print(f"  {'-'*40} {'-'*6} {'-'*16} {'-'*10}")
    for i in range(n_actuators):
        name = model.actuator(i).name
        gear = model.actuator_gear[i, 0]
        frc_lo = model.actuator_forcerange[i, 0]
        frc_hi = model.actuator_forcerange[i, 1]
        gain = model.actuator_gainprm[i, 0]
        print(f"  {name:<40} {gear:>6.0f} [{frc_lo:>6.0f}, {frc_hi:>6.0f}] {gain:>10.1f}")


# ---------------------------------------------------------------------------
# 4. Observation / Action space
# ---------------------------------------------------------------------------
def check_spaces(env):
    _header("4. Observation & Action Space")
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    act_low = env.action_space.low
    act_high = env.action_space.high

    print(f"  Observation shape: {obs_shape}")
    print(f"  Action shape:      {act_shape}")
    print(f"  Action range:      [{act_low.min():.1f}, {act_high.max():.1f}]")

    # Check for very large observation spaces
    obs_dim = obs_shape[0] if len(obs_shape) == 1 else np.prod(obs_shape)
    if obs_dim > 1000:
        _warn(f"Large observation space ({obs_dim} dims). May slow learning — "
              "consider feature selection or normalization.")
    else:
        _ok(f"Observation dimensionality ({obs_dim}) is manageable")

    # Check observation values at reset
    obs, _ = env.reset()
    if np.any(np.isnan(obs)):
        _fail("NaN in initial observation")
    elif np.any(np.abs(obs) > 1000):
        _warn(f"Large observation values detected (max abs={np.abs(obs).max():.1f}). "
              "Consider observation normalization (VecNormalize).")
    else:
        _ok(f"Observation values at reset are bounded (max abs={np.abs(obs).max():.2f})")


# ---------------------------------------------------------------------------
# 5. Reward signal quality
# ---------------------------------------------------------------------------
def check_reward_signal(env):
    _header("5. Reward Signal Quality")
    env.reset()

    # Collect rewards under random actions
    n_episodes = 5
    episode_rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        env.reset()
        ep_reward = 0.0
        ep_len = 0
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1
            if terminated or truncated:
                break
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)

    mean_r = np.mean(episode_rewards)
    std_r = np.std(episode_rewards)
    mean_len = np.mean(episode_lengths)

    print(f"  Random policy ({n_episodes} episodes):")
    print(f"    Mean episode reward: {mean_r:.2f} +/- {std_r:.2f}")
    print(f"    Mean episode length: {mean_len:.1f} steps")

    if mean_len < 5:
        _fail("Episodes terminate almost immediately under random actions. "
              "The model may be too unstable or termination conditions too strict.")
    elif mean_len < 20:
        _warn(f"Short episodes (mean {mean_len:.0f} steps). Sparse signal — "
              "consider relaxing termination or adding curriculum.")
    else:
        _ok(f"Episodes survive {mean_len:.0f} steps on average under random policy")

    # Check reward at standing pose (zero-action from target)
    env.reset()
    obs, reward, _, _, _ = env.step(np.zeros(env.action_space.shape))
    print(f"  Reward at target pose (zero action, 1 step): {reward:.4f}")

    if reward <= 0:
        _warn("Reward at target pose is non-positive. The standing pose "
              "should yield the highest reward.")
    else:
        _ok(f"Positive reward at target pose ({reward:.4f})")


# ---------------------------------------------------------------------------
# 6. Gravity compensation check
# ---------------------------------------------------------------------------
def check_gravity_compensation(env):
    _header("6. Gravity Compensation Feasibility")
    model = env.unwrapped.mj_model
    data = env.unwrapped.mj_data
    env.reset()

    # Compute gravity forces using mj_inverse
    data_copy = mujoco.MjData(model)
    data_copy.qpos[:] = data.qpos[:]
    data_copy.qvel[:] = data.qvel[:]
    mujoco.mj_forward(model, data_copy)

    # Zero out velocities and accelerations
    data_copy.qvel[:] = 0
    data_copy.qacc[:] = 0
    mujoco.mj_inverse(model, data_copy)

    # qfrc_inverse gives the forces needed to maintain current pose
    gravity_forces = data_copy.qfrc_inverse.copy()

    print(f"  Gravity compensation forces (qfrc_inverse):")
    print(f"    Norm: {np.linalg.norm(gravity_forces):.2f}")
    print(f"    Max abs: {np.abs(gravity_forces).max():.2f}")

    # Check if actuators can produce enough force
    # For each actuated DOF, check if the max actuator force covers the gravity load
    n_act = model.nu
    underactuated_joints = []

    for i in range(n_act):
        # Get the joint this actuator drives
        joint_id = model.actuator_trnid[i, 0]
        dof_adr = model.jnt_dofadr[joint_id]
        jnt_type = model.jnt_type[joint_id]

        if jnt_type == 3:  # Hinge
            grav_torque = abs(gravity_forces[dof_adr])
            gear = model.actuator_gear[i, 0]
            gain = model.actuator_gainprm[i, 0]
            frc_range = model.actuator_forcerange[i]

            # Max force = gain * gear * ctrl (ctrl in [-1,1])
            # But force is also clamped by forcerange
            max_force = min(abs(gain * gear), abs(frc_range[1]))

            if grav_torque > max_force * 0.8:
                underactuated_joints.append(
                    (model.actuator(i).name, grav_torque, max_force))

    if underactuated_joints:
        _warn(f"{len(underactuated_joints)} actuators may struggle against gravity:")
        for name, grav, maxf in underactuated_joints:
            pct = (grav / maxf * 100) if maxf > 0 else float('inf')
            print(f"    - {name}: gravity={grav:.1f}, max_force={maxf:.1f} "
                  f"({pct:.0f}% of capacity)")
    else:
        _ok("All actuators have sufficient authority to counter gravity")

    # Check unactuated joints under gravity
    actuated_dofs = set()
    for i in range(n_act):
        joint_id = model.actuator_trnid[i, 0]
        dof_adr = model.jnt_dofadr[joint_id]
        jnt_type = model.jnt_type[joint_id]
        if jnt_type == 3:
            actuated_dofs.add(dof_adr)
        elif jnt_type == 0:  # Free joint
            for d in range(6):
                actuated_dofs.add(dof_adr + d)

    passive_under_gravity = []
    for i in range(model.njnt):
        dof_adr = model.jnt_dofadr[i]
        jnt_type = model.jnt_type[i]
        if jnt_type == 3 and dof_adr not in actuated_dofs:
            grav_torque = abs(gravity_forces[dof_adr])
            stiffness = model.jnt_stiffness[i]
            damping = model.dof_damping[dof_adr]
            if grav_torque > 0.1:
                passive_under_gravity.append(
                    (model.joint(i).name, grav_torque, stiffness, damping))

    if passive_under_gravity:
        print(f"\n  Unactuated joints with significant gravity load:")
        print(f"  {'Joint':<35} {'Gravity':>8} {'Stiffness':>10} {'Damping':>8}")
        print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*8}")
        for name, grav, stiff, damp in sorted(
                passive_under_gravity, key=lambda x: -x[1])[:15]:
            print(f"  {name:<35} {grav:>8.2f} {stiff:>10.1f} {damp:>8.1f}")
        if len(passive_under_gravity) > 15:
            print(f"  ... and {len(passive_under_gravity) - 15} more")

        heavily_loaded = [x for x in passive_under_gravity if x[1] > x[2]]
        if heavily_loaded:
            _warn(f"{len(heavily_loaded)} unactuated joints have gravity loads "
                  "exceeding their passive stiffness. These joints will drift "
                  "and cannot be controlled by the policy.")


# ---------------------------------------------------------------------------
# 7. Reference motion playback
# ---------------------------------------------------------------------------
def check_reference_playback(env):
    _header("7. Reference Motion Playback")
    unwrapped = env.unwrapped

    if not hasattr(unwrapped, 'target_qpos'):
        _warn("No target_qpos found — skipping reference check")
        return

    target = unwrapped.target_qpos
    model = unwrapped.mj_model
    data = unwrapped.mj_data

    # Set model to target pose and check for penetrations / instability
    data_test = mujoco.MjData(model)
    data_test.qpos[:] = target
    data_test.qvel[:] = 0
    mujoco.mj_forward(model, data_test)

    # Check contact forces at reference pose
    n_contacts = data_test.ncon
    print(f"  Contacts at reference pose: {n_contacts}")
    if n_contacts > 0:
        contact_forces = []
        for c in range(n_contacts):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data_test, c, force)
            contact_forces.append(np.linalg.norm(force))
        max_cf = max(contact_forces)
        print(f"  Max contact force: {max_cf:.2f}")
        if max_cf > 1000:
            _warn("Large contact forces at reference pose — bodies may be "
                  "interpenetrating. Check mesh collision geometry.")
        else:
            _ok(f"Contact forces are reasonable (max={max_cf:.2f})")

    # Check if reference pose is within joint limits
    out_of_limits = []
    for i in range(model.njnt):
        if not model.jnt_limited[i]:
            continue
        q_adr = model.jnt_qposadr[i]
        jnt_type = model.jnt_type[i]
        if jnt_type == 3:  # Hinge
            val = target[q_adr]
            lo = model.jnt_range[i, 0]
            hi = model.jnt_range[i, 1]
            if val < lo - 0.01 or val > hi + 0.01:
                out_of_limits.append(
                    (model.joint(i).name, val, lo, hi))

    if out_of_limits:
        _fail(f"{len(out_of_limits)} joints exceed their limits in reference pose:")
        for name, val, lo, hi in out_of_limits:
            print(f"    - {name}: value={val:.4f}, range=[{lo:.4f}, {hi:.4f}]")
    else:
        _ok("Reference pose is within all joint limits")

    # Simulate forward from reference pose with zero action
    env.reset()
    initial_qpos = data.qpos.copy()
    steps = 100
    for _ in range(steps):
        env.step(np.zeros(env.action_space.shape))
    drift = np.linalg.norm(data.qpos - initial_qpos)
    print(f"  Pose drift after {steps} zero-action steps: {drift:.4f}")
    if drift > 5.0:
        _warn("Large drift from reference under zero action — the reference "
              "pose is not a stable equilibrium. Consider increasing joint "
              "stiffness/damping or adding a PD controller layer.")
    else:
        _ok(f"Drift from reference is moderate ({drift:.4f})")


# ---------------------------------------------------------------------------
# 8. Simulation parameters summary
# ---------------------------------------------------------------------------
def check_simulation_params(env):
    _header("8. Simulation Parameters")
    model = env.unwrapped.mj_model

    timestep = model.opt.timestep
    frame_skip = env.unwrapped.frame_skip if hasattr(env.unwrapped, 'frame_skip') else 'N/A'
    integrator_names = {0: 'Euler', 1: 'implicit', 2: 'implicitfast', 3: 'RK4'}
    integrator = integrator_names.get(model.opt.integrator, str(model.opt.integrator))

    print(f"  Timestep:     {timestep}")
    print(f"  Frame skip:   {frame_skip}")
    if isinstance(frame_skip, int):
        print(f"  Control dt:   {timestep * frame_skip:.4f}s ({1/(timestep*frame_skip):.0f} Hz)")
    print(f"  Integrator:   {integrator}")
    print(f"  Gravity:      {model.opt.gravity.tolist()}")

    # Joint defaults
    stiffnesses = model.jnt_stiffness
    dampings = model.dof_damping
    armatures = model.dof_armature

    print(f"\n  Joint stiffness:  min={stiffnesses.min():.2f}, "
          f"max={stiffnesses.max():.2f}, mean={stiffnesses.mean():.2f}")
    print(f"  DOF damping:      min={dampings.min():.2f}, "
          f"max={dampings.max():.2f}, mean={dampings.mean():.2f}")
    print(f"  DOF armature:     min={armatures.min():.4f}, "
          f"max={armatures.max():.4f}, mean={armatures.mean():.4f}")

    if timestep > 0.005:
        _warn(f"Large timestep ({timestep}s). With complex contact dynamics, "
              "consider reducing to 0.001-0.002s.")
    if integrator == 'Euler':
        _warn("Using Euler integrator. For musculoskeletal models with stiff "
              "tendons/contacts, 'implicitfast' may be more stable.")
    if dampings.min() < 0.1:
        _warn(f"Some DOFs have very low damping ({dampings.min():.4f}). "
              "This can cause oscillations.")
    if armatures.min() < 0.01:
        _warn(f"Some DOFs have very low armature ({armatures.min():.4f}). "
              "Low armature can cause numerical instability in contact-rich "
              "scenarios.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    env_name = "MyoSkeletonStand-v0"
    if len(sys.argv) > 1:
        env_name = sys.argv[1]

    print(f"MyoSkeleton Model Diagnostic — Environment: {env_name}")
    print(f"{'='*60}")

    env = check_loading(env_name)
    if env is None:
        print("\nCannot proceed — environment failed to load.")
        sys.exit(1)

    check_simulation_params(env)
    check_spaces(env)
    check_stability(env)
    check_actuator_authority(env)
    check_reward_signal(env)
    check_gravity_compensation(env)
    check_reference_playback(env)

    _header("SUMMARY")
    print("  Review all [WARN] and [FAIL] items above.")
    print("  Common fixes for unstable musculoskeletal models:")
    print("    - Increase joint damping (default class in XML)")
    print("    - Add armature to reduce numerical stiffness")
    print("    - Reduce simulation timestep")
    print("    - Switch integrator to 'implicitfast'")
    print("    - Increase actuator gear/gain for underactuated joints")
    print("    - Add actuators for critical unactuated joints")
    print("    - Relax termination conditions during early training")
    print("    - Use observation normalization (VecNormalize)")
    print()

    env.close()


if __name__ == "__main__":
    main()
