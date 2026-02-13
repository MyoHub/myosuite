"""Convert myoskeleton H5 motion data to mjlab NPZ format.

This script reads a soccer1.h5 motion file (from myosuite), plays it back
through the myoskeleton MuJoCo model (with fingers disabled), and saves the
resulting trajectories in the NPZ format expected by mjlab's MotionLoader:

  - joint_pos:      (T, num_actuated_joints)  joint positions
  - joint_vel:      (T, num_actuated_joints)  joint velocities
  - body_pos_w:     (T, num_bodies, 3)        world-frame body positions
  - body_quat_w:    (T, num_bodies, 4)        world-frame body quaternions (wxyz)
  - body_lin_vel_w: (T, num_bodies, 3)        world-frame body linear velocities
  - body_ang_vel_w: (T, num_bodies, 3)        world-frame body angular velocities

Usage:
    python convert_h5_to_npz.py
"""

from __future__ import annotations

import os

import h5py
import mujoco
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
H5_FILE = os.path.join(
    os.path.dirname(__file__), "myosuite", "envs", "myo", "mjx", "soccer1.h5"
)
XML_FILE = os.path.join(
    os.path.dirname(__file__),
    "myosuite",
    "simhive",
    "myo_model",
    "myoskeleton",
    "myoskeleton.xml",
)
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "soccer1_myoskeleton.npz")

# ── Finger joints to disable ────────────────────────────────────────────────
FINGER_JOINTS = [
    "cmc_flexion_r", "cmc_abduction_r", "mp_flexion_r", "ip_flexion_r",
    "mcp2_flexion_r", "mcp2_abduction_r", "pm2_flexion_r", "md2_flexion_r",
    "mcp3_flexion_r", "mcp3_abduction_r", "pm3_flexion_r", "md3_flexion_r",
    "mcp4_flexion_r", "mcp4_abduction_r", "pm4_flexion_r", "md4_flexion_r",
    "mcp5_flexion_r", "mcp5_abduction_r", "pm5_flexion_r", "md5_flexion_r",
    "cmc_flexion_l", "cmc_abduction_l", "mp_flexion_l", "ip_flexion_l",
    "mcp2_flexion_l", "mcp2_abduction_l", "pm2_flexion_l", "md2_flexion_l",
    "mcp3_flexion_l", "mcp3_abduction_l", "pm3_flexion_l", "md3_flexion_l",
    "mcp4_flexion_l", "mcp4_abduction_l", "pm4_flexion_l", "md4_flexion_l",
    "mcp5_flexion_l", "mcp5_abduction_l", "pm5_flexion_l", "md5_flexion_l",
]


def load_h5_motion(h5_path: str) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load H5 motion file and return (time, qpos_dict, qvel_dict)."""
    with h5py.File(h5_path, "r") as f:
        motion_name = list(f.keys())[0]
        grp = f[motion_name]
        time = np.array(grp["time"], dtype=np.float64)

        qpos_dict: dict[str, np.ndarray] = {}
        for key in grp["qpos"]:
            qpos_dict[key] = np.array(grp["qpos"][key], dtype=np.float64).squeeze()

        qvel_dict: dict[str, np.ndarray] = {}
        if "qvel" in grp:
            for key in grp["qvel"]:
                qvel_dict[key] = np.array(grp["qvel"][key], dtype=np.float64).squeeze()

    return time, qpos_dict, qvel_dict


def build_model_from_xml(xml_path: str) -> mujoco.MjModel:
    """Load the raw myoskeleton model from XML (no spec modifications)."""
    return mujoco.MjModel.from_xml_path(xml_path)


def get_joint_names(model: mujoco.MjModel) -> list[str]:
    """Return all joint names in model order."""
    return [model.joint(i).name for i in range(model.njnt)]


def get_actuated_joint_names(model: mujoco.MjModel) -> list[str]:
    """Return joint names that are NOT the free root and NOT finger joints."""
    all_names = get_joint_names(model)
    return [
        n for n in all_names
        if n != "myoskeleton_root" and n not in FINGER_JOINTS
    ]


def main() -> None:
    print(f"Loading H5 from {H5_FILE}")
    time_arr, qpos_dict, qvel_dict = load_h5_motion(H5_FILE)
    horizon = len(time_arr)
    print(f"  Motion: {horizon} timesteps")

    print(f"Loading model from {XML_FILE}")
    model = build_model_from_xml(XML_FILE)
    data = mujoco.MjData(model)

    all_joint_names = get_joint_names(model)
    actuated_joints = get_actuated_joint_names(model)
    print(f"  Total joints: {model.njnt}, actuated (no fingers): {len(actuated_joints)}")

    # Joint names present in both the h5 and the model (excluding root & fingers).
    h5_joint_names = [k for k in qpos_dict.keys() if k != "myoskeleton_root"]
    valid_joints = [j for j in h5_joint_names if j in all_joint_names]

    # ── Allocate output arrays ───────────────────────────────────────────────
    num_bodies = model.nbody
    joint_pos_all = np.zeros((horizon, len(actuated_joints)), dtype=np.float32)
    joint_vel_all = np.zeros((horizon, len(actuated_joints)), dtype=np.float32)
    body_pos_w = np.zeros((horizon, num_bodies, 3), dtype=np.float32)
    body_quat_w = np.zeros((horizon, num_bodies, 4), dtype=np.float32)
    body_lin_vel_w = np.zeros((horizon, num_bodies, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((horizon, num_bodies, 3), dtype=np.float32)

    # ── Playback ─────────────────────────────────────────────────────────────
    print("Playing back motion through model ...")
    dt = float(model.opt.timestep)

    for t in range(horizon):
        # Set root pos/quat.
        if "myoskeleton_root" in qpos_dict:
            root_data = qpos_dict["myoskeleton_root"]
            if root_data.ndim == 1:
                data.qpos[:7] = root_data
            else:
                data.qpos[:7] = root_data[t]

        # Set joint positions from h5.
        for jn in valid_joints:
            idx = all_joint_names.index(jn)
            qposadr = model.jnt_qposadr[idx]
            val = qpos_dict[jn]
            data.qpos[qposadr] = val[t] if val.ndim > 0 and val.shape[0] == horizon else val

        # Forward kinematics to get body positions.
        mujoco.mj_forward(model, data)

        # Store body positions and quaternions.
        body_pos_w[t] = data.xpos.copy()
        body_quat_w[t] = data.xquat.copy()

        # Store actuated joint positions (excluding free root and fingers).
        for j_idx, jn in enumerate(actuated_joints):
            if jn in all_joint_names:
                model_idx = all_joint_names.index(jn)
                qposadr = model.jnt_qposadr[model_idx]
                joint_pos_all[t, j_idx] = data.qpos[qposadr]

    # ── Compute velocities via finite differences ────────────────────────────
    print("Computing velocities via finite differences ...")
    dt_motion = float(np.mean(np.diff(time_arr))) if horizon > 1 else dt

    # Body linear velocity: d(pos)/dt
    for t in range(horizon):
        if t == 0:
            body_lin_vel_w[t] = (body_pos_w[1] - body_pos_w[0]) / dt_motion if horizon > 1 else 0.0
        elif t == horizon - 1:
            body_lin_vel_w[t] = (body_pos_w[t] - body_pos_w[t - 1]) / dt_motion
        else:
            body_lin_vel_w[t] = (body_pos_w[t + 1] - body_pos_w[t - 1]) / (2 * dt_motion)

    # Body angular velocity from quaternion differences.
    for t in range(horizon):
        if t == 0:
            t0, t1 = 0, min(1, horizon - 1)
        elif t == horizon - 1:
            t0, t1 = t - 1, t
        else:
            t0, t1 = t - 1, t + 1
        delta_t = (t1 - t0) * dt_motion
        if delta_t < 1e-12:
            continue
        for b in range(num_bodies):
            q0 = body_quat_w[t0, b]  # wxyz
            q1 = body_quat_w[t1, b]
            # Ensure same hemisphere.
            if np.dot(q0, q1) < 0:
                q1 = -q1
            # dq = q1 * q0_inv; angular velocity = 2 * dq.xyz / dt
            q0_inv = np.array([q0[0], -q0[1], -q0[2], -q0[3]])
            q0_inv /= np.dot(q0, q0) + 1e-12
            dq_w = q0_inv[0] * q1[0] - q0_inv[1] * q1[1] - q0_inv[2] * q1[2] - q0_inv[3] * q1[3]
            dq_x = q0_inv[0] * q1[1] + q0_inv[1] * q1[0] + q0_inv[2] * q1[3] - q0_inv[3] * q1[2]
            dq_y = q0_inv[0] * q1[2] - q0_inv[1] * q1[3] + q0_inv[2] * q1[0] + q0_inv[3] * q1[1]
            dq_z = q0_inv[0] * q1[3] + q0_inv[1] * q1[2] - q0_inv[2] * q1[1] + q0_inv[3] * q1[0]
            body_ang_vel_w[t, b] = 2.0 * np.array([dq_x, dq_y, dq_z]) / delta_t

    # Joint velocity via finite differences.
    for t in range(horizon):
        if t == 0:
            joint_vel_all[t] = (joint_pos_all[1] - joint_pos_all[0]) / dt_motion if horizon > 1 else 0.0
        elif t == horizon - 1:
            joint_vel_all[t] = (joint_pos_all[t] - joint_pos_all[t - 1]) / dt_motion
        else:
            joint_vel_all[t] = (joint_pos_all[t + 1] - joint_pos_all[t - 1]) / (2 * dt_motion)

    # NOTE: h5 qvel data is often zero-valued (float16 precision loss), so we
    # always use finite-differenced velocities computed above.

    # ── Save NPZ ─────────────────────────────────────────────────────────────
    print(f"Saving to {OUTPUT_FILE}")
    print(f"  joint_pos:      {joint_pos_all.shape}")
    print(f"  joint_vel:      {joint_vel_all.shape}")
    print(f"  body_pos_w:     {body_pos_w.shape}")
    print(f"  body_quat_w:    {body_quat_w.shape}")
    print(f"  body_lin_vel_w: {body_lin_vel_w.shape}")
    print(f"  body_ang_vel_w: {body_ang_vel_w.shape}")
    print(f"  actuated_joints: {actuated_joints[:5]} ... ({len(actuated_joints)} total)")

    np.savez(
        OUTPUT_FILE,
        joint_pos=joint_pos_all,
        joint_vel=joint_vel_all,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    print("Done.")


if __name__ == "__main__":
    main()
