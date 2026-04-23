import jax.numpy as jp
import jax

# Constants for floating-point precision
_FLOAT_EPS = jp.finfo(jp.float32).eps
_EPS4 = _FLOAT_EPS * 4.0


def mulQuat(qa, qb):
    res = jp.zeros(4)
    res = res.at[0].set(qa[0] * qb[0] - qa[1] * qb[1] - qa[2] * qb[2] - qa[3] * qb[3])
    res = res.at[1].set(qa[0] * qb[1] + qa[1] * qb[0] + qa[2] * qb[3] - qa[3] * qb[2])
    res = res.at[2].set(qa[0] * qb[2] - qa[1] * qb[3] + qa[2] * qb[0] + qa[3] * qb[1])
    res = res.at[3].set(qa[0] * qb[3] + qa[1] * qb[2] - qa[2] * qb[1] + qa[3] * qb[0])
    return res


def negQuat(quat):
    return jp.array([quat[0], -quat[1], -quat[2], -quat[3]])


def quat2Vel(quat, dt=1):
    axis = quat[1:].copy()
    sin_a_2 = jp.sqrt(jp.sum(axis**2))
    axis = axis / (sin_a_2 + 1e-8)
    speed = 2 * jp.arctan2(sin_a_2, quat[0]) / dt
    return speed, axis


def diffQuat(quat1, quat2):
    neg = negQuat(quat1)
    return mulQuat(quat2, neg)


def quatDiff2Vel(quat1, quat2, dt):
    diff = diffQuat(quat1, quat2)
    return quat2Vel(diff, dt)


def axis_angle2quat(axis, angle):
    c = jp.cos(angle / 2)
    s = jp.sin(angle / 2)
    return jp.array([c, s * axis[0], s * axis[1], s * axis[2]])


def euler2mat(euler):
    euler = jp.asarray(euler, dtype=jp.float32)
    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = jp.sin(ai), jp.sin(aj), jp.sin(ak)
    ci, cj, ck = jp.cos(ai), jp.cos(aj), jp.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = jp.empty(euler.shape[:-1] + (3, 3), dtype=jp.float32)
    mat = mat.at[..., 2, 2].set(cj * ck)
    mat = mat.at[..., 2, 1].set(sj * sc - cs)
    mat = mat.at[..., 2, 0].set(sj * cc + ss)
    mat = mat.at[..., 1, 2].set(cj * sk)
    mat = mat.at[..., 1, 1].set(sj * ss + cc)
    mat = mat.at[..., 1, 0].set(sj * cs - sc)
    mat = mat.at[..., 0, 2].set(-sj)
    mat = mat.at[..., 0, 1].set(cj * si)
    mat = mat.at[..., 0, 0].set(cj * ci)
    return mat


def euler2quat(euler):
    euler = jp.asarray(euler, dtype=jp.float32)
    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = jp.sin(ai), jp.sin(aj), jp.sin(ak)
    ci, cj, ck = jp.cos(ai), jp.cos(aj), jp.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = jp.empty(euler.shape[:-1] + (4,), dtype=jp.float32)
    quat = quat.at[..., 0].set(cj * cc + sj * ss)
    quat = quat.at[..., 3].set(cj * sc - sj * cs)
    quat = quat.at[..., 2].set(-(cj * ss + sj * cc))
    quat = quat.at[..., 1].set(cj * cs - sj * sc)
    return quat


def mat2euler(mat):
    mat = jp.asarray(mat, dtype=jp.float32)
    cy = jp.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = jp.empty(mat.shape[:-2] + (3,), dtype=jp.float32)
    euler = euler.at[..., 2].set(
        jp.where(
            condition,
            -jp.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
            -jp.arctan2(-mat[..., 1, 0], mat[..., 1, 1]),
        )
    )
    euler = euler.at[..., 1].set(
        jp.where(
            condition,
            -jp.arctan2(-mat[..., 0, 2], cy),
            -jp.arctan2(-mat[..., 0, 2], cy),
        )
    )
    euler = euler.at[..., 0].set(
        jp.where(condition, -jp.arctan2(mat[..., 1, 2], mat[..., 2, 2]), 0.0)
    )
    return euler


def mat2quat(mat):
    """Convert Rotation Matrix to Quaternion using JAX"""
    mat = jp.asarray(mat, dtype=jp.float32)
    assert mat.shape == (3, 3), f"Invalid shape matrix {mat.shape}"

    def case_1(mat):
        trace = 1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]
        s = 2.0 * jp.sqrt(trace)
        s = jp.where(mat[1, 2] < mat[2, 1], -s, s)
        q1 = 0.25 * s
        s = 1.0 / s
        q0 = (mat[1, 2] - mat[2, 1]) * s
        q2 = (mat[0, 1] + mat[1, 0]) * s
        q3 = (mat[2, 0] + mat[0, 2]) * s
        return jp.array([q0, q1, q2, q3])

    def case_2(mat):
        trace = 1.0 - mat[0, 0] + mat[1, 1] - mat[2, 2]
        s = 2.0 * jp.sqrt(trace)
        s = jp.where(mat[2, 0] < mat[0, 2], -s, s)
        q2 = 0.25 * s
        s = 1.0 / s
        q0 = (mat[2, 0] - mat[0, 2]) * s
        q1 = (mat[0, 1] + mat[1, 0]) * s
        q3 = (mat[1, 2] + mat[2, 1]) * s
        return jp.array([q0, q1, q2, q3])

    def case_3(mat):
        trace = 1.0 - mat[0, 0] - mat[1, 1] + mat[2, 2]
        s = 2.0 * jp.sqrt(trace)
        s = jp.where(mat[0, 1] < mat[1, 0], -s, s)
        q3 = 0.25 * s
        s = 1.0 / s
        q0 = (mat[0, 1] - mat[1, 0]) * s
        q1 = (mat[2, 0] + mat[0, 2]) * s
        q2 = (mat[1, 2] + mat[2, 1]) * s
        return jp.array([q0, q1, q2, q3])

    def case_4(mat):
        trace = 1.0 + mat[0, 0] + mat[1, 1] + mat[2, 2]
        s = 2.0 * jp.sqrt(trace)
        q0 = 0.25 * s
        s = 1.0 / s
        q1 = (mat[1, 2] - mat[2, 1]) * s
        q2 = (mat[2, 0] - mat[0, 2]) * s
        q3 = (mat[0, 1] - mat[1, 0]) * s
        return jp.array([q0, q1, q2, q3])

    # Conditional execution for efficiency
    q = jax.lax.cond(
        mat[2, 2] < 0.0,
        lambda mat: jax.lax.cond(mat[0, 0] > mat[1, 1], case_1, case_2, mat),
        lambda mat: jax.lax.cond(mat[0, 0] < -mat[1, 1], case_3, case_4, mat),
        mat,
    )

    q = q.at[1:].set(-q[1:])
    return q


def quat2euler(quat):
    return mat2euler(quat2mat(quat))


def quat2mat(quat):
    quat = jp.asarray(quat, dtype=jp.float32)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = jp.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = jp.empty(quat.shape[:-1] + (3, 3), dtype=jp.float32)
    mat = mat.at[..., 0, 0].set(1.0 - (yY + zZ))
    mat = mat.at[..., 0, 1].set(xY - wZ)
    mat = mat.at[..., 0, 2].set(xZ + wY)
    mat = mat.at[..., 1, 0].set(xY + wZ)
    mat = mat.at[..., 1, 1].set(1.0 - (xX + zZ))
    mat = mat.at[..., 1, 2].set(yZ - wX)
    mat = mat.at[..., 2, 0].set(xZ - wY)
    mat = mat.at[..., 2, 1].set(yZ + wX)
    mat = mat.at[..., 2, 2].set(1.0 - (xX + yY))
    return jp.where((Nq > _FLOAT_EPS)[..., jp.newaxis, jp.newaxis], mat, jp.eye(3))


def rotVecMatT(vec, mat):
    return jp.array(
        [
            mat[0, 0] * vec[0] + mat[1, 0] * vec[1] + mat[2, 0] * vec[2],
            mat[0, 1] * vec[0] + mat[1, 1] * vec[1] + mat[2, 1] * vec[2],
            mat[0, 2] * vec[0] + mat[1, 2] * vec[1] + mat[2, 2] * vec[2],
        ]
    )


def rotVecMat(vec, mat):
    return jp.array(
        [
            mat[0, 0] * vec[0] + mat[0, 1] * vec[1] + mat[0, 2] * vec[2],
            mat[1, 0] * vec[0] + mat[1, 1] * vec[1] + mat[1, 2] * vec[2],
            mat[2, 0] * vec[0] + mat[2, 1] * vec[1] + mat[2, 2] * vec[2],
        ]
    )


def rotVecQuat(vec, quat):
    mat = quat2mat(quat)
    return rotVecMat(vec, mat)


def quat2euler_intrinsic(quat):
    w, x, y, z = quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = jp.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = jp.where(jp.abs(sinp) >= 1, jp.copysign(jp.pi / 2, sinp), jp.arcsin(sinp))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = jp.arctan2(siny_cosp, cosy_cosp)

    return jp.array([roll, pitch, yaw])


def intrinsic_euler2quat(euler):
    roll, pitch, yaw = euler
    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5

    sin_roll = jp.sin(half_roll)
    cos_roll = jp.cos(half_roll)
    sin_pitch = jp.sin(half_pitch)
    cos_pitch = jp.cos(half_pitch)
    sin_yaw = jp.sin(half_yaw)
    cos_yaw = jp.cos(half_yaw)

    w = cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw
    x = sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw
    y = cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw
    z = cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw

    return jp.array([w, x, y, z])
