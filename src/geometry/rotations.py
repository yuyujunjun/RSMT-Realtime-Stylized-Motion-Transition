from pytorch3d.transforms import rotation_6d_to_matrix, quaternion_to_matrix
import torch.nn.functional as F
from src.geometry.vector import cross_product, normalize_vector
import torch
import numpy as np

# m: batch*3*3
# out: batch*4*4
def get_4x4_rotation_matrix_from_3x3_rotation_matrix(m):
    batch_size = m.shape[0]

    row4 = torch.autograd.Variable(torch.zeros(batch_size, 1, 3).type_as(m))
    m43 = torch.cat((m, row4), 1)  # batch*4,3
    col4 = torch.autograd.Variable(torch.zeros(batch_size, 4, 1).type_as(m))
    col4[:, 3, 0] = col4[:, 3, 0] + 1
    out = torch.cat((m43, col4), 2)  # batch*4*4

    return out


# matrices batch*3*3
# both matrix are orthogonal rotation matrices
# out theta between 0 to 180 degree batch
def compute_geodesic_distance_from_two_matrices(m1, m2):
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

    theta = compute_angle_from_rotation_matrix(m)
    # theta = torch.min(theta, 2*np.pi - theta)

    return theta


# matrices batch*3*3
# both matrix are orthogonal rotation matrices
# out theta between 0 to 180 degree batch
def compute_angle_from_rotation_matrix(m):
    batch = m.shape[0]

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).type_as(m)))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).type_as(m)) * -1)

    theta = torch.acos(cos)

    return theta


# axis: size (batch, 3)
# output quat order is w, x, y, z
def get_random_rotation_around_axis(axis, return_quaternion=False):
    batch = axis.shape[0]
    axis = normalize_vector(axis)  # batch*3
    theta = torch.FloatTensor(axis.shape[0]).uniform_(-np.pi, np.pi).type_as(axis)  # [0, pi] #[-180, 180]
    sin = torch.sin(theta)
    qw = torch.cos(theta)
    qx = axis[:, 0] * sin
    qy = axis[:, 1] * sin
    qz = axis[:, 2] * sin

    quaternion = torch.cat((qw.view(batch, 1), qx.view(batch, 1), qy.view(batch, 1), qz.view(batch, 1)), 1)
    matrix = quaternion_to_matrix(quaternion)

    if (return_quaternion == True):
        return matrix, quaternion
    else:
        return matrix


# axisAngle batch*4 angle, x,y,z
# output quat order is w, x, y, z
def get_random_rotation_matrices_around_random_axis(batch, return_quaternion=False):
    axis = torch.autograd.Variable(torch.randn(batch.shape[0], 3).type_as(batch))
    return get_random_rotation_around_axis(axis, return_quaternion=return_quaternion)


# matrices batch*3*3
# both matrix are orthogonal rotation matrices
# out theta between 0 to 180 degree batch
def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).type_as(m1)))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).type_as(m1)) * -1)

    theta = torch.acos(cos)

    # theta = torch.min(theta, 2*np.pi - theta)

    return theta


def geodesic_loss(gt_r_matrix, out_r_matrix):
    theta = compute_geodesic_distance_from_two_matrices(gt_r_matrix, out_r_matrix)
    error = theta.mean()
    return error


def geodesic_loss_matrix3x3_matrix3x3(gt_r_matrix, out_r_matrix):
    return geodesic_loss(gt_r_matrix, out_r_matrix)


def geodesic_loss_quat_ortho6d(quaternions, ortho6d):

    assert quaternions.shape[-1] == 4, "quaternions should have the last dimension length be 4"
    assert ortho6d.shape[-1] == 6, "ortho6d should have the last dimension length be 6"

    # quat -> matrix3x3
    matrix3x3_a = quaternion_to_matrix(quaternions)

    # ortho6d -> matrix3x3
    matrix3x3_b = rotation_6d_to_matrix(ortho6d)

    return geodesic_loss_matrix3x3_matrix3x3(matrix3x3_a, matrix3x3_b)


def rotation_6d_to_matrix_no_cross(d6: torch.Tensor) -> torch.Tensor:
    """
    This is the pytorch3d implementation of the conversion,
    but avoids the torch.cross() operator that cannot be exported with opset_version >= 11
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = cross_product(b1, b2)
    return torch.stack((b1, b2, b3), dim=-2)
