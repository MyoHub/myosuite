import os
import time
from pathlib import Path

import click
import numpy as np

from myosuite.utils import gym

# Try to import motion file utilities
try:
    from myosuite.envs.myo.myotracking.download_motion_data import (
        download_default_motion,
        read_mot,
    )
except ImportError:
    read_mot = None
    download_default_motion = None

# Try to import video saving
try:
    import skvideo.io
except ImportError:
    skvideo = None

DESC = """
Script to render trajectories embedded in the env or from external motion files.

Supports:
- Built-in reference trajectories (env.playback())
- External motion files (.npz, .mot)
- Onscreen and offscreen rendering
- Video saving
"""  # noqa: E501


@click.command(help=DESC)
@click.option(
    "-e",
    "--env_name",
    type=str,
    help="environment to load",
    default="MyoHandBananaPass-v0",
)
@click.option("-h", "--horizon", type=int, help="playback horizon", default=-1)
@click.option(
    "-n",
    "--num_playback",
    type=int,
    help="Number of time to loop playback",
    default=1,
)
@click.option(
    "-r",
    "--render",
    type=click.Choice(["onscreen", "offscreen", "none"]),
    help="visualize onscreen, offscreen, or none",
    default="onscreen",
)
@click.option(
    "-m",
    "--motion_file",
    type=str,
    default=None,
    help=(
        "Path to external motion file (.npz or .mot). "
        "If provided, uses external motion instead of built-in reference."
    ),
)
@click.option(
    "-o",
    "--output_file",
    type=str,
    default=None,
    help="Output video file path (required for offscreen rendering)",
)
@click.option(
    "-c",
    "--camera_name",
    type=str,
    default=None,
    help="Camera name for rendering",
)
@click.option(
    "-fs",
    "--frame_size",
    type=tuple,
    default=(640, 480),
    help="Frame size (width, height) for video",
)
@click.option(
    "--fps",
    type=float,
    default=30.0,
    help="Frames per second for video",
)
@click.option(
    "--task",
    type=str,
    default=None,
    help=(
        "Task type for Kinesis/LocoMuJoCo environments " "(e.g., 'motion_imitation')"
    ),
)
def examine_reference(
    env_name,
    horizon,
    num_playback,
    render,
    motion_file,
    output_file,
    camera_name,
    frame_size,
    fps,
    task,
):
    # Create environment
    if task:
        # Try to create Kinesis/LocoMuJoCo environment
        try:
            if task == "motion_imitation":
                from myosuite.envs.myo.myotracking.kinesis import (
                    create_myosuite_kinesis_env,
                )

                env = create_myosuite_kinesis_env(
                    task=task,
                    motion_file=motion_file,
                    max_episode_length=1000,
                )
            elif task in ["walk", "run", "humanoid"]:
                from myosuite.envs.myo.myotracking.locomujoco import (
                    create_myo_locomujoco_env,
                )

                env = create_myo_locomujoco_env(task=task, loco_mujoco_compatible=True)
            else:
                env = gym.make(env_name)
        except ImportError:
            msg = (
                f"Warning: Could not import task-specific environment, "
                f"using {env_name}"
            )
            print(msg)
            env = gym.make(env_name)
    else:
        env = gym.make(env_name)

    # Handle external motion file
    if motion_file:
        # Download default motion if not provided
        if motion_file == "default" and download_default_motion:
            try:
                motion_file = download_default_motion()
                print(f"Using downloaded motion file: {motion_file}")
            except Exception as e:
                print(f"Warning: Could not download motion: {e}")
                motion_file = None

        if motion_file and os.path.exists(motion_file):
            # Use external motion file playback
            _playback_external_motion(
                env=env,
                motion_file=motion_file,
                num_playback=num_playback,
                render=render,
                output_file=output_file,
                camera_name=camera_name,
                frame_size=frame_size,
                fps=fps,
            )
            if hasattr(env, "close"):
                env.close()
            return

    # Use built-in reference motion (original functionality)
    # fixed or random reference
    if horizon == 1:
        horizon = env.spec.max_episode_steps

    # infer reference horizon
    env_unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    if horizon == -1:
        if hasattr(env_unwrapped, "ref") and hasattr(env_unwrapped.ref, "horizon"):
            horizon = env_unwrapped.ref.horizon
        elif hasattr(env, "spec") and hasattr(env.spec, "max_episode_steps"):
            horizon = env.spec.max_episode_steps
        else:
            horizon = 1000

    # Check if environment has playback method
    if not hasattr(env_unwrapped, "playback"):
        msg = (
            f"Warning: Environment {env_name} does not have playback() "
            "method. Cannot playback built-in reference motion."
        )
        print(msg)
        if hasattr(env, "close"):
            env.close()
        return

    # Prepare for offscreen rendering
    frames = []
    if render == "offscreen":
        if not hasattr(env, "mj_renderer"):
            print(
                "Error: Environment does not support offscreen rendering. "
                "Switching to onscreen rendering."
            )
            render = "onscreen"
        elif output_file is None:
            output_file = f"reference_motion_{env_name}.mp4"

    # Start playback loops
    print(f"Rendering reference motion (total frames: {horizon})")
    for n in range(num_playback):
        print(f"Playback loop: {n + 1}/{num_playback}")
        env.reset()

        # Rollout a traj
        for h in range(horizon):
            has_more = env_unwrapped.playback()

            # render
            if render == "onscreen":
                env.mj_render()
                time.sleep(env_unwrapped.dt)
            elif render == "offscreen":
                if camera_name:
                    frame = env.mj_renderer.render_camera_offscreen(
                        cameras=[camera_name],
                        width=frame_size[0],
                        height=frame_size[1],
                        device_id=0,
                    )
                    frames.append(frame[0])
                else:
                    frame = env.mj_renderer.render_offscreen(
                        width=frame_size[0],
                        height=frame_size[1],
                        camera_id=0,
                    )
                    frames.append(frame)

            if not has_more:
                break

    # Save video if offscreen rendering
    if render == "offscreen" and frames and output_file:
        _save_video(frames, output_file, fps)

    if hasattr(env, "close"):
        env.close()


def _playback_external_motion(
    env,
    motion_file,
    num_playback,
    render,
    output_file,
    camera_name,
    frame_size,
    fps,
):
    """Playback external motion file."""
    # Unwrap environment to access base environment
    env_unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    # Try to get base_env if it's a custom wrapper
    base_env = getattr(env_unwrapped, "base_env", None)
    if base_env is not None:
        # Use base_env for MuJoCo access
        mj_env = base_env.unwrapped if hasattr(base_env, "unwrapped") else base_env
    else:
        mj_env = env_unwrapped

    # Load motion data
    motion_data, joint_names = _load_motion_data(motion_file)

    # Get joint indices in environment
    if hasattr(mj_env, "mj_model"):
        env_joint_names = [
            mj_env.mj_model.joint(i).name for i in range(mj_env.mj_model.njnt)
        ]
    else:
        env_joint_names = []

    # Map motion joints to environment joints
    joint_indices = _map_joints_to_env(joint_names, env_joint_names)

    # Prepare rendering - check for renderer in wrapped or unwrapped env
    frames = []
    if render == "offscreen":
        # Check for renderer in wrapped env first, then unwrapped
        if hasattr(env, "mj_renderer"):
            mj_renderer = env.mj_renderer
        elif hasattr(env_unwrapped, "mj_renderer"):
            mj_renderer = env_unwrapped.mj_renderer
        elif base_env and hasattr(base_env, "mj_renderer"):
            mj_renderer = base_env.mj_renderer
        else:
            print(
                "Error: Environment does not support offscreen rendering. "
                "Switching to onscreen rendering."
            )
            render = "onscreen"
            mj_renderer = None
        if output_file is None:
            output_file = f"motion_playback_{Path(motion_file).stem}.mp4"
    else:
        mj_renderer = None

    print(f"Playing back {len(motion_data)} frames from {motion_file}...")
    for n in range(num_playback):
        print(f"Playback loop: {n + 1}/{num_playback}")
        obs = env.reset()
        if isinstance(obs, tuple):
            obs, _ = obs

        for frame_idx, motion_frame in enumerate(motion_data):
            # Convert to radians if needed
            if np.max(np.abs(motion_frame)) > np.pi:
                motion_frame = np.deg2rad(motion_frame)

            # Set joint positions from motion data
            if hasattr(mj_env, "mj_data") and hasattr(mj_env, "mj_model"):
                try:
                    import mujoco
                except ImportError:
                    print("Warning: mujoco not available")
                    break

                for i, joint_idx in enumerate(joint_indices):
                    if joint_idx is not None and i < len(motion_frame):
                        try:
                            qpos_adr = mj_env.mj_model.joint(joint_idx).qposadr
                            if len(qpos_adr) > 0:
                                mj_env.mj_data.qpos[qpos_adr[0]] = motion_frame[i]
                        except (IndexError, AttributeError):
                            pass

                # Forward kinematics
                mujoco.mj_forward(mj_env.mj_model, mj_env.mj_data)

            # Render
            if render == "onscreen":
                # Try to call mj_render on unwrapped env
                if hasattr(mj_env, "mj_render"):
                    mj_env.mj_render()
                elif hasattr(env, "render"):
                    env.render()
                else:
                    # Skip rendering if not available
                    pass

                dt = getattr(mj_env, "dt", 0.01)
                time.sleep(dt)
            elif render == "offscreen" and mj_renderer:
                if camera_name:
                    frame = mj_renderer.render_camera_offscreen(
                        cameras=[camera_name],
                        width=frame_size[0],
                        height=frame_size[1],
                        device_id=0,
                    )
                    frames.append(frame[0])
                else:
                    frame = mj_renderer.render_offscreen(
                        width=frame_size[0],
                        height=frame_size[1],
                        camera_id=0,
                    )
                    frames.append(frame)

    # Save video if offscreen rendering
    if render == "offscreen" and frames and output_file:
        _save_video(frames, output_file, fps)
        print(f"Video saved to: {output_file}")


def _load_motion_data(motion_file):
    """Load motion data from file."""
    motion_file = Path(motion_file)

    if motion_file.suffix == ".npz":
        data = np.load(motion_file, allow_pickle=True)
        if "motion" in data:
            motion_data = data["motion"]
        else:
            keys = list(data.keys())
            if len(keys) > 0:
                motion_data = data[keys[0]]
            else:
                raise ValueError(f"No motion data found in {motion_file}")

        if "joint_names" in data:
            joint_names = data["joint_names"].tolist()
        else:
            joint_names = [f"joint_{i}" for i in range(motion_data.shape[1])]

    elif motion_file.suffix == ".mot":
        if read_mot is None:
            raise ImportError(
                "read_mot function not available. "
                "Ensure download_motion_data.py is accessible."
            )
        df = read_mot(str(motion_file))
        if "time" in df.columns:
            df = df.drop(columns=["time"])
        motion_data = df.values
        joint_names = df.columns.tolist()
    else:
        raise ValueError(
            f"Unsupported file format: {motion_file.suffix}. "
            "Supported formats: .npz, .mot"
        )

    return motion_data, joint_names


def _map_joints_to_env(motion_joint_names, env_joint_names):
    """Map motion joint names to environment joint indices."""
    joint_indices = []
    for name in motion_joint_names:
        if name in env_joint_names:
            joint_indices.append(env_joint_names.index(name))
        else:
            # Try partial match
            matches = [
                i
                for i, env_name in enumerate(env_joint_names)
                if name in env_name or env_name in name
            ]
            if matches:
                joint_indices.append(matches[0])
            else:
                joint_indices.append(None)
    return joint_indices


def _save_video(frames, output_file, fps):
    """Save frames as video file."""
    if skvideo is None:
        print(
            "Warning: skvideo not available. Cannot save video. "
            "Install with: pip install scikit-video"
        )
        return

    frames_array = np.asarray(frames)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    import platform

    if platform.system() == "Darwin":
        skvideo.io.vwrite(
            str(output_file),
            frames_array,
            outputdict={"-pix_fmt": "yuv420p"},
        )
    else:
        skvideo.io.vwrite(str(output_file), frames_array)

    print(f"Saved {len(frames)} frames to {output_file}")


if __name__ == "__main__":
    examine_reference()
