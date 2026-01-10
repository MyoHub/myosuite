import numpy as np
from jax.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R


def quat_scalarfirst2scalarlast(quat):
    """
    Converts a quaternion from scalar-first to scalar-last representation.
    """
    return quat[..., [1, 2, 3, 0]]


def calculate_relative_site_quatities(
    data, rel_site_ids, rel_body_ids, body_rootid, backend
):

    if backend == np:
        R = np_R
    else:
        R = jnp_R

    # get site positions and rotations
    site_xpos_traj = data.site_xpos
    site_xmat_traj = data.site_xmat
    site_xpos_traj = site_xpos_traj[rel_site_ids]
    site_xmat_traj = site_xmat_traj[rel_site_ids]

    # get relevant properties and calculate site velocities
    main_site_id = 0  # --> zeroth index in rel_site_ids
    site_root_body_id = body_rootid[rel_body_ids]
    site_xvel = calc_site_velocities(
        rel_site_ids, data, rel_body_ids, site_root_body_id, backend
    )
    main_site_xvel = site_xvel[main_site_id]
    site_xvel = backend.delete(site_xvel, main_site_id, axis=0)

    # calculate the rotation matrix from main site to the other sites
    main_site_xmat_traj = site_xmat_traj[main_site_id].reshape(3, 3)
    site_xmat_traj = backend.delete(site_xmat_traj, main_site_id, axis=0).reshape(
        -1, 3, 3
    )
    rel_rot_mat = calculate_relative_rotation_matrices(
        main_site_xmat_traj, site_xmat_traj, backend
    )

    # calculate relative quantities
    main_site_xpos_traj = site_xpos_traj[main_site_id]
    site_xpos_traj = backend.delete(site_xpos_traj, main_site_id, axis=0)
    site_rpos = calc_rel_positions(site_xpos_traj, main_site_xpos_traj, backend)
    site_rangles = R.from_matrix(rel_rot_mat).as_rotvec()
    site_rvel = calculate_relative_velocity_in_local_frame(
        main_site_xvel, site_xvel, main_site_xmat_traj, rel_rot_mat, backend
    )

    return site_rpos, site_rangles, site_rvel
