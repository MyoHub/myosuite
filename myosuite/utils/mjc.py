"""Utility helpers for querying MuJoCo model element IDs by name.

These thin wrappers call ``mujoco.mj_name2id`` with the appropriate
``mujoco.mjtObj`` type, providing clearer, typed entry points at callsites.
"""

import mujoco


def body_name2id(model, name):
    """Return the integer ID of a body by name.

    Args:
        model: A ``mujoco.MjModel`` instance.
        name: The body name (string) to resolve.

    Returns:
        The integer ID for the specified body, or -1 if not found.
    """
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def joint_name2id(model, name):
    """Return the integer ID of a joint by name.

    Args:
        model: A ``mujoco.MjModel`` instance.
        name: The joint name (string) to resolve.

    Returns:
        The integer ID for the specified joint, or -1 if not found.
    """
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def actuator_name2id(model, name):
    """Return the integer ID of an actuator by name.

    Args:
        model: A ``mujoco.MjModel`` instance.
        name: The actuator name (string) to resolve.

    Returns:
        The integer ID for the specified actuator, or -1 if not found.
    """
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def sensor_name2id(model, name):
    """Return the integer ID of a sensor by name.

    Args:
        model: A ``mujoco.MjModel`` instance.
        name: The sensor name (string) to resolve.

    Returns:
        The integer ID for the specified sensor, or -1 if not found.
    """
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)


def site_name2id(model, name):
    """Return the integer ID of a site by name.

    Args:
        model: A ``mujoco.MjModel`` instance.
        name: The site name (string) to resolve.

    Returns:
        The integer ID for the specified site, or -1 if not found.
    """
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def geom_name2id(model, name):
    """Return the integer ID of a geom by name.

    Args:
        model: A ``mujoco.MjModel`` instance.
        name: The geom name (string) to resolve.

    Returns:
        The integer ID for the specified geom, or -1 if not found.
    """
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
