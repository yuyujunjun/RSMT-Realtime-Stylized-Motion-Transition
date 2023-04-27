import torch
import torch.nn.functional as F
from src.geometry.vector import normalize_vector
import numpy as np
import pytorch3d.transforms
def rotation_6d_to_matrix_no_normalized(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    # b1 = F.normalize(a1, dim=-1)
    # b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    # b2 = F.normalize(b2, dim=-1)
    # b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((a1, a2), dim=-2)

def rotation6d_multiply(r1,r2):
    r1 = pytorch3d.transforms.rotation_6d_to_matrix(r1)
    r2 = pytorch3d.transforms.rotation_6d_to_matrix(r2)
    return pytorch3d.transforms.matrix_to_rotation_6d(torch.matmul(r1,r2))
def rotation6d_apply(r,p):
    #p :...3
    r = pytorch3d.transforms.rotation_6d_to_matrix(r)
    p = p.unsqueeze(-1)#...3,1
    return torch.matmul(r,p).squeeze(-1)
def rotation6d_inverse(r):
    r = pytorch3d.transforms.rotation_6d_to_matrix(r)
    inv_r = torch.transpose(r,-2,-1)
    return pytorch3d.transforms.matrix_to_rotation_6d(inv_r)
def quat_to_or6D(quat):

    assert(quat.shape[-1]==4)
    return pytorch3d.transforms.matrix_to_rotation_6d(pytorch3d.transforms.quaternion_to_matrix(quat))
def or6d_to_quat(mat):
    assert (mat.shape[-1] == 6)
    return pytorch3d.transforms.matrix_to_quaternion(pytorch3d.transforms.rotation_6d_to_matrix(mat))
def normalized_or6d(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    return torch.cat((b1, b2), dim=-1)
#移除四元数不连续的问题
def remove_quat_discontinuities(rotations):
    rotations = rotations.clone()
    rots_inv = -rotations
    for i in range(1, rotations.shape[1]):
        replace_mask = torch.sum(rotations[:, i-1:i, ...] * rotations[:, i:i+1, ...], 
                                 dim=-1, keepdim=True) < \
                       torch.sum(rotations[:, i-1:i, ...] * rots_inv[:, i:i+1, ...], 
                                 dim=-1, keepdim=True)
        replace_mask = replace_mask.squeeze(1).type_as(rotations)
        rotations[:, i, ...] = replace_mask * rots_inv[:, i, ...] + (1.0 - replace_mask) * rotations[:, i, ...]
    return rotations

def from_to_1_0_0(v_from):
    '''v_from: B,3'''
    v_to_unit = torch.tensor(np.array([[1, 0, 0]]), dtype=v_from.dtype, device=v_from.device).expand(v_from.shape)
    y = torch.tensor(np.array([[0,1,0]]),dtype=v_from.dtype,device=v_from.device).expand(v_from.shape)
    z_to_unit = torch.tensor(np.array([[0,0,1]]),dtype=v_from.dtype,device=v_from.device).expand(v_from.shape)

    v_from_unit = normalize_vector(v_from)
    z_from_unit = normalize_vector(torch.cross(v_from_unit,y,dim=-1))
    to_transform = torch.stack([v_to_unit,y,z_to_unit],dim=-1)
    from_transform = torch.stack([v_from_unit,y,z_from_unit],dim=-1)
    shape = to_transform.shape
    to_transform = to_transform.view(-1,3,3)
    from_transform = from_transform.view(-1,3,3)
    r = torch.matmul(to_transform,from_transform.transpose(1,2)).view(shape)
    rq = pytorch3d.transforms.matrix_to_quaternion(r)

    # w = (v_from_unit * v_to_unit).sum(dim=1) + 1
    # '''can't cross if two directions are exactly inverse'''
    # xyz = torch.cross(v_from_unit, v_to_unit, dim=1)
    # '''if exactly inverse, the rotation should be (0,0,1,0), around yaxis 180'''
    # xyz[...,1] = torch.where(w==0,torch.tensor([1],dtype=v_from.dtype,device=xyz.device),xyz[...,1])
    # q = torch.cat([w.unsqueeze(1), xyz], dim=1)
    return rq


# returns quaternion so that v_from rotated by this quaternion equals v_to
# v_... are vectors of size (..., 3)
# returns quaternion in w, x, y, z order, of size (..., 4)
# note: such a rotation is not unique, there is an infinite number of solutions
# this implementation returns the shortest arc
def from_to_quaternion(v_from, v_to):
    v_from_unit = normalize_vector(v_from)
    v_to_unit = normalize_vector(v_to)

    w = (v_from_unit * v_to_unit).sum(dim=-1)+1
    '''can't cross if two directions are exactly inverse'''
    xyz = torch.cross(v_from_unit, v_to_unit, dim=-1)


    q = torch.cat([w.unsqueeze(-1), xyz], dim=-1)
    return normalize_vector(q)

def quat_inv(quat):
    return pytorch3d.transforms.quaternion_invert(quat)
def quat_mul(quat0,quat1):
    return pytorch3d.transforms.quaternion_multiply(quat0,quat1)
def quat_mul_vec(quat,vec):
    return pytorch3d.transforms.quaternion_apply(quat,vec)
def slerp(q0, q1, t):
    """
    Spherical Linear Interpolation of quaternions
    https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/slerp/index.htm
    :param q0: Start quats (w, x, y, z) : shape = (B, J, 4)
    :param q1: End quats (w, x, y, z) : shape = (B, J, 4)
    :param t:  Step (in [0, 1]) : shape = (B, J, 1)
    :return: Interpolated quat (w, x, y, z) : shape = (B, J, 4)
    """
  #  q0 = q0.unsqueeze(1)
  #  q1 = q1.unsqueeze(1)
    
    # Dot product
    q = q0*q1
    cos_half_theta = torch.sum(q, dim=-1, keepdim=True)
   # t = t.view(1,-1,1,1)
    # Make sure we take the shortest path :
    q1_antipodal = -q1
    q1 = torch.where(cos_half_theta < 0, q1_antipodal, q1)
    cos_half_theta = torch.where(cos_half_theta < 0,-cos_half_theta,cos_half_theta)
    half_theta = torch.acos(cos_half_theta)
    # torch.sin must be safer here
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)
    ratio_a = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratio_b = torch.sin(t * half_theta) / sin_half_theta
    
    qt = ratio_a * q0 + ratio_b * q1    
    # If the angle was constant, prevent nans by picking the original quat:
    qt = torch.where(torch.abs(cos_half_theta) >= 1.0-1e-8, q0, qt)
    return qt
