import numpy
import numpy as np
import pytorch3d.transforms
_FLOAT_EPS = np.finfo(np.float32).eps
_EPS4 = _FLOAT_EPS * 4.0
_EPS16 = _FLOAT_EPS * 16.0
def clamp_mean(x,axis):
    x = np.asarray(x,dtype=np.float64)
    return np.mean(x,axis=axis)
def clamp_std(x,axis):
    x = np.asarray(x,dtype=np.float64)
    std = np.std(x,axis=axis)
    return np.where(np.abs(std) < _EPS4, _EPS4, std)
def length(x, axis=-1, keepdims=True):
    """
    Computes vector norm along a tensor axis(axes)

    :param x: tensor
    :param axis: axis(axes) along which to compute the norm
    :param keepdims: indicates if the dimension(s) on axis should be kept
    :return: The length or vector of lengths.
    """
    lgth = np.sqrt(np.sum(x * x, axis=axis, keepdims=keepdims))
    return lgth


def normalize(x, axis=-1):
    """
    Normalizes a tensor over some axis (axes)

    :param x: data tensor
    :param axis: axis(axes) along which to compute the norm
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized tensor
    """
    x = np.array(x,dtype=np.float64)
    res = x / (length(x, axis=axis) + _EPS4)
    return res


def quat_normalize(x):
    """
    Normalizes a quaternion tensor

    :param x: data tensor
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized quaternions tensor
    """
    res = normalize(x)
    return res


def angle_axis_to_quat(angle, axis):
    """
    Converts from and angle-axis representation to a quaternion representation

    :param angle: angles tensor
    :param axis: axis tensor
    :return: quaternion tensor
    """
    c = np.cos(angle / 2.0)[..., np.newaxis]
    s = np.sin(angle / 2.0)[..., np.newaxis]
    q = np.concatenate([c, s * axis], axis=-1)
    return q


def euler_to_quat(e, order='zyx'):
    """

    Converts from an euler representation to a quaternion representation

    :param e: euler tensor
    :param order: order of euler rotations
    :return: quaternion tensor
    """
    axis = {
        'x': np.asarray([1, 0, 0], dtype=np.float32),
        'y': np.asarray([0, 1, 0], dtype=np.float32),
        'z': np.asarray([0, 0, 1], dtype=np.float32)}

    q0 = angle_axis_to_quat(e[..., 0], axis[order[0]])
    q1 = angle_axis_to_quat(e[..., 1], axis[order[1]])
    q2 = angle_axis_to_quat(e[..., 2], axis[order[2]])

    return quat_mul(q0, quat_mul(q1, q2))


def quat_inv(q):
    """
    Inverts a tensor of quaternions

    :param q: quaternion tensor
    :return: tensor of inverted quaternions
    """
    res = np.asarray([1, -1, -1, -1], dtype=np.float32) * q
    return res


# def quat_fk(lrot, lpos, parents):
#     """
#     Performs Forward Kinematics (FK) on local quaternions and local positions to retrieve global representations
#
#     :param lrot: tensor of local quaternions with shape (..., Nb of joints, 4)
#     :param lpos: tensor of local positions with shape (..., Nb of joints, 3)
#     :param parents: list of parents indices
#     :return: tuple of tensors of global quaternion, global positions
#     """
#     gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
#     for i in range(1, len(parents)):
#         gp.append(quat_mul_vec(gr[parents[i]], lpos[..., i:i+1, :]) + gp[parents[i]])
#         gr.append(quat_mul    (gr[parents[i]], lrot[..., i:i+1, :]))
#
#     res = np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)
#     return res
#
#
# def quat_ik(grot, gpos, parents):
#     """
#     Performs Inverse Kinematics (IK) on global quaternions and global positions to retrieve local representations
#
#     :param grot: tensor of global quaternions with shape (..., Nb of joints, 4)
#     :param gpos: tensor of global positions with shape (..., Nb of joints, 3)
#     :param parents: list of parents indices
#     :return: tuple of tensors of local quaternion, local positions
#     """
#     res = [
#         np.concatenate([
#             grot[..., :1, :],
#             quat_mul(quat_inv(grot[..., parents[1:], :]), grot[..., 1:, :]),
#         ], axis=-2),
#         np.concatenate([
#             gpos[..., :1, :],
#             quat_mul_vec(
#                 quat_inv(grot[..., parents[1:], :]),
#                 gpos[..., 1:, :] - gpos[..., parents[1:], :]),
#         ], axis=-2)
#     ]
#
#     return res


def quat_mul(x, y):
    """
    Performs quaternion multiplication on arrays of quaternions

    :param x: tensor of quaternions of shape (..., Nb of joints, 4)
    :param y: tensor of quaternions of shape (..., Nb of joints, 4)
    :return: The resulting quaternions
    """
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    res = np.concatenate([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], axis=-1)

    return res


def quat_mul_vec(q, x):
    """
    Performs multiplication of an array of 3D vectors by an array of quaternions (rotation).

    :param q: tensor of quaternions of shape (..., Nb of joints, 4)
    :param x: tensor of vectors of shape (..., Nb of joints, 3)
    :return: the resulting array of rotated vectors
    """
    t = 2.0 * np.cross(q[..., 1:], x)
    res = x + q[..., 0][..., np.newaxis] * t + np.cross(q[..., 1:], t)

    return res


def quat_slerp(x, y, a):
    """
    Perfroms spherical linear interpolation (SLERP) between x and y, with proportion a

    :param x: quaternion tensor
    :param y: quaternion tensor
    :param a: indicator (between 0 and 1) of completion of the interpolation.
    :return: tensor of interpolation results
    """
    len = np.sum(x * y, axis=-1)

    neg = len < 0.0
    len[neg] = -len[neg]
    y[neg] = -y[neg]

    a = np.zeros_like(x[..., 0]) + a
    amount0 = np.zeros(a.shape)
    amount1 = np.zeros(a.shape)

    linear = (1.0 - len) < 0.01
    omegas = np.arccos(len[~linear])
    sinoms = np.sin(omegas)

    amount0[linear] = 1.0 - a[linear]
    amount0[~linear] = np.sin((1.0 - a[~linear]) * omegas) / sinoms

    amount1[linear] = a[linear]
    amount1[~linear] = np.sin(a[~linear] * omegas) / sinoms
    res = amount0[..., np.newaxis] * x + amount1[..., np.newaxis] * y

    return res


def quat_between(x, y):
    """
    Quaternion rotations between two 3D-vector arrays

    :param x: tensor of 3D vectors
    :param y: tensor of 3D vetcors
    :return: tensor of quaternions
    """
    res = np.concatenate([
        np.sqrt(np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1))[..., np.newaxis] +
        np.sum(x * y, axis=-1)[..., np.newaxis],
        np.cross(x, y)], axis=-1)

    return res


def interpolate_local(lcl_r_mb,lcl_q_mb,length,target):
    # Extract last past frame and target frame
    start_lcl_r_mb = lcl_r_mb[:, 10 - 1, :, :][:, None, :, :]  # (B, 1, J, 3)
    end_lcl_r_mb = lcl_r_mb[:, target, :, :][:, None, :, :]
    start_lcl_q_mb = lcl_q_mb[:, 10 - 1, :, :]
    end_lcl_q_mb = lcl_q_mb[:, target, :, :]
    # LERP Local Positions:
    n_trans = length
    interp_ws = np.linspace(0.0, 1.0, num=n_trans + 2, dtype=np.float32)
    offset = end_lcl_r_mb - start_lcl_r_mb
    const_trans = np.tile(start_lcl_r_mb, [1, n_trans + 2, 1, 1])
    inter_lcl_r_mb = const_trans + (interp_ws)[None, :, None, None] * offset

    # SLERP Local Quats:
    interp_ws = np.linspace(0.0, 1.0, num=n_trans + 2, dtype=np.float32)
    inter_lcl_q_mb = np.stack(
        [(quat_normalize(quat_slerp(quat_normalize(start_lcl_q_mb), quat_normalize(end_lcl_q_mb), w))) for w in
         interp_ws], axis=1)

    return inter_lcl_r_mb, inter_lcl_q_mb
def interpolate_local_(lcl_r_mb, lcl_q_mb, n_past, n_future):
    """
    Performs interpolation between 2 frames of an animation sequence.

    The 2 frames are indirectly specified through n_past and n_future.
    SLERP is performed on the quaternions
    LERP is performed on the root's positions.

    :param lcl_r_mb:  Local/Global root positions (B, T, 1, 3)
    :param lcl_q_mb:  Local quaternions (B, T, J, 4)
    :param n_past:    Number of frames of past context
    :param n_future:  Number of frames of future context
    :return: Interpolated root and quats
    """

    # Extract last past frame and target frame
    start_lcl_r_mb = lcl_r_mb[:, n_past - 1, :, :][:, None, :, :]  # (B, 1, J, 3)
    end_lcl_r_mb = lcl_r_mb[:, -n_future, :, :][:, None, :, :]

    start_lcl_q_mb = lcl_q_mb[:, n_past - 1, :, :]
    end_lcl_q_mb = lcl_q_mb[:, -n_future, :, :]

    # LERP Local Positions:
    n_trans = lcl_r_mb.shape[1] - (n_past + n_future)
    interp_ws = np.linspace(0.0, 1.0, num=n_trans + 2, dtype=np.float32)
    offset = end_lcl_r_mb - start_lcl_r_mb

    const_trans    = np.tile(start_lcl_r_mb, [1, n_trans + 2, 1, 1])
    inter_lcl_r_mb = const_trans + (interp_ws)[None, :, None, None] * offset

    # SLERP Local Quats:
    interp_ws = np.linspace(0.0, 1.0, num=n_trans + 2, dtype=np.float32)
    inter_lcl_q_mb = np.stack(
        [(quat_normalize(quat_slerp(quat_normalize(start_lcl_q_mb), quat_normalize(end_lcl_q_mb), w))) for w in
         interp_ws], axis=1)

    return inter_lcl_r_mb, inter_lcl_q_mb

import torch
def remove_quat_discontinuities(rotations:torch.tensor):
    """

    Removing quat discontinuities on the time dimension (removing flips)

    :param rotations: Array of quaternions of shape (T, J, 4)
    :return: The processed array without quaternion inversion.
    """
    rotations = rotations.clone()
    rots_inv = -rotations
    for i in range(1, rotations.shape[1]):
        replace_mask = torch.sum(rotations[:, i - 1:i, ...] * rotations[:, i:i + 1, ...],
                                 dim=-1, keepdim=True) < \
                       torch.sum(rotations[:, i - 1:i, ...] * rots_inv[:, i:i + 1, ...],
                                 dim=-1, keepdim=True)
        replace_mask = replace_mask.squeeze(1).type_as(rotations)
        rotations[:, i, ...] = replace_mask * rots_inv[:, i, ...] + (1.0 - replace_mask) * rotations[:, i, ...]
    return rotations


# Orient the data according to the las past keframe
# def rotate_at_frame(X, Q, parents, n_past=10):
#     """
#     Re-orients the animation data according to the last frame of past context.
#
#     :param X: tensor of local positions of shape (Batchsize, Timesteps, Joints, 3)
#     :param Q: tensor of local quaternions (Batchsize, Timesteps, Joints, 4)
#     :param parents: list of parents' indices
#     :param n_past: number of frames in the past context
#     :return: The rotated positions X and quaternions Q
#     """
#     # Get global quats and global poses (FK)
#     global_q, global_x = quat_fk(Q, X, parents)
#
#     key_glob_Q = global_q[:, n_past - 1: n_past, 0:1, :]  # (B, 1, 1, 4)
#     forward = np.array([1, 0, 1])[np.newaxis, np.newaxis, np.newaxis, :] \
#                  * quat_mul_vec(key_glob_Q, np.array([0, 1, 0])[np.newaxis, np.newaxis, np.newaxis, :])
#     forward = normalize(forward)
#     yrot = quat_normalize(quat_between(np.array([1, 0, 0]), forward))#1,0,0
#     new_glob_Q = quat_mul(quat_inv(yrot), global_q)
#     new_glob_X = quat_mul_vec(quat_inv(yrot), global_x)
#
#     # back to local quat-pos
#     Q, X = quat_ik(new_glob_Q, new_glob_X, parents)
#
#     return X, Q


def extract_feet_contacts(pos, lfoot_idx, rfoot_idx, velfactor=0.02):
    """
    Extracts binary tensors of feet contacts

    :param pos: tensor of global positions of shape (Timesteps, Joints, 3)
    :param lfoot_idx: indices list of left foot joints
    :param rfoot_idx: indices list of right foot joints
    :param velfactor: velocity threshold to consider a joint moving or not
    :return: binary tensors of left foot contacts and right foot contacts
    """
    lfoot_xyz = (pos[1:, lfoot_idx, :] - pos[:-1, lfoot_idx, :]) ** 2
    contacts_l = (np.sqrt(np.sum(lfoot_xyz, axis=-1)) < velfactor)

    rfoot_xyz = (pos[1:, rfoot_idx, :] - pos[:-1, rfoot_idx, :]) ** 2
    contacts_r = (np.sqrt(np.sum(rfoot_xyz, axis=-1)) < velfactor)

    # Duplicate the last frame for shape consistency
    contacts_l = np.concatenate([contacts_l, contacts_l[-1:]], axis=0)
    contacts_r = np.concatenate([contacts_r, contacts_r[-1:]], axis=0)

    return contacts_l, contacts_r

##########
# My own code
#########
# axis sequences for Euler angles

def quat_to_mat(quat):

    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat.shape)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))
def mat_to_euler(mat):
    """ Convert Rotation Matrix to Euler Angles.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(condition,
                             -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                             -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
    euler[..., 1] = np.where(condition,
                             -np.arctan2(-mat[..., 0, 2], cy),
                             -np.arctan2(-mat[..., 0, 2], cy))
    euler[..., 0] = np.where(condition,
                             -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
                             0.0)
    return euler
def quat_to_euler(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    return mat_to_euler(quat_to_mat(quat))

