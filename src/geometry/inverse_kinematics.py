from typing import List

import torch
import numpy as np
from pytorch3d.transforms import quaternion_multiply, quaternion_apply, quaternion_invert

# performs differentiable forward kinematics
# local_transforms: local transform matrix of each transform (B, J, 4, 4)
# level_transforms: a list of hierarchy levels, ordered by distance to the root transforms in the hierarchy
#                   each elements of the list is a list transform indexes
#                   for instance level_transforms[3] contains a list of indices of all the transforms that are 3 levels
#                   below the root transforms the indices should match these of the provided local transforms
#                   for instance if level_transforms[4, 1] is equal to 7, it means that local_transform[:, 7, :, :]
#                   contains the local matrix of a transform that is 4 levels deeper than the root transform
#                   (ie there are 4 transforms above it in the hierarchy)
# level_transform_parents: similar to level_transforms but contains the parent indices
#                   for instance if level_transform_parents[4, 1] is equal to 5, it means that local_transform[:, 5, :, :]
#                   contains the local matrix of the parent transform of the transform contained in level_transforms[4, 1]
# out: world transform matrix of each transform in the batch (B, J, 4, 4). Order is the same as input


def invert_transform_hierarchy(global_transforms: torch.Tensor, level_transforms: List[List[int]],
                               level_transform_parents: List[List[int]]):
    # used to store local transforms
    local_transforms = global_transforms.clone()

    # then process all children transforms
    for level in range(len(level_transforms)-1, 0, -1):
        parent_bone_indices = level_transform_parents[level]
        local_bone_indices = level_transforms[level]
        parent_level_transforms = global_transforms[..., parent_bone_indices, :, :]
        local_level_transforms = global_transforms[..., local_bone_indices, :, :]
        local_matrix = torch.matmul(torch.inverse(parent_level_transforms),
                                    local_level_transforms)
        local_transforms[..., local_bone_indices, :, :] = local_matrix.type_as(local_transforms)

    return local_transforms


# Compute a transform matrix combining a rotation and translation (rotation is applied first)
# translations: translation vectors (N, 3)
# rotations: rotation matrices (N, 3, 3)
# out: transform matrix (N, 4, 4)
def get_transform_matrix(translations: torch.Tensor, rotations: torch.Tensor):    
    out_shape = np.array(rotations.shape)
    out_shape[-2:] = 4    
    out = torch.zeros(list(out_shape)).type_as(rotations)
    
    out[..., 0:3, 0:3] = rotations
    out[..., 0:3, 3] = translations
    out[..., 3, 3] = 1.0
    return out


# performs differentiable inverse kinematics
# global_rotations: global rotation matrices of each joint in the batch (B, J, 3, 3)
# global_offset: local position offset of each joint in the batch (B, J, 3).
#               This corresponds to the offset with respect to the parent joint when no rotation is applied
# level_joints: a list of hierarchy levels, ordered by distance to the root joints in the hierarchy
#               each elements of the list is a list transform indexes
#               for instance level_joints[3] contains a list of indices of all the joints that are 3 levels
#               below the root joints the indices should match these of the provided local rotations
#               for instance if level_joints[4, 1] is equal to 7, it means that local_rotations[:, 7, :, :]
#               contains the local rotation matrix of a joint that is 4 levels deeper than the root joint
#               (ie there are 4 joints above it in the hierarchy)
# level_joint_parents: similar to level_transforms but contains the parent indices
#                      for instance if level_joint_parents[4, 1] is equal to 5, it means that local_rotations[:, 5, :, :]
#                      contains the local rotation matrix of the parent joint of the joint contained in level_joints[4, 1]
def inverse_kinematics(global_rotations: torch.Tensor, global_offsets: torch.Tensor,
                       level_joints: List[List[int]], level_joint_parents: List[List[int]]):

    # compute global transformation matrix for each joints
    global_transforms = get_transform_matrix(translations=global_offsets, rotations=global_rotations)
    # used to store local transforms
    local_transforms = global_transforms.clone()
    
    # then process all children transforms
    for level in range(len(level_joints)-1, 0, -1):
        parent_bone_indices = level_joint_parents[level]
        local_bone_indices = level_joints[level]
        parent_level_transforms = global_transforms[..., parent_bone_indices, :, :]
        local_level_transforms = global_transforms[..., local_bone_indices, :, :]
        local_matrix = torch.matmul(torch.inverse(parent_level_transforms),
                                    local_level_transforms)
        local_transforms[..., local_bone_indices, :, :] = local_matrix.type_as(local_transforms)

    return local_transforms

def inverse_kinematics_only_quats(joints_global_rotations: torch.Tensor, parents):
    """
    Performs differentiable inverse kinematics with quaternion-based rotations
    :param joints_global_rotations: tensor of global quaternions of shape (..., J, 4)
    :param joints_global_positions: tensor of global positions of shape (..., J, 3)
    :param a list of hierarchy levels, ordered by distance to the root joints in the hierarchy
                           each elements of the list is a list transform indexes
                           for instance level_joints[3] contains a list of indices of all the joints that are 3 levels
                           below the root joints the indices should match these of the provided local rotations
                           for instance if level_joints[4, 1] is equal to 7, it means that local_rotations[:, 7, :, :]
                           contains the local rotation matrix of a joint that is 4 levels deeper than the root joint
                           (ie there are 4 joints above it in the hierarchy)
    :param level_joint_parents: similar to level_transforms but contains the parent indices
                      for instance if level_joint_parents[4, 1] is equal to 5, it means that local_rotations[:, 5, :, :]
                      contains the local rotation matrix of the parent joint of the joint contained in level_joints[4, 1]
    :return:
        joints_local_positions: tensor of local bone offsets of shape (..., J, 3)
        joints_local_rotations: tensor of local quaternions of shape (..., J, 4)
    """


    joint_local_quats = joints_global_rotations.clone()
    invert_global_rotations = quaternion_invert(joint_local_quats)
    joint_local_quats = torch.cat((
        joint_local_quats[..., :1, :],
        quaternion_multiply(invert_global_rotations[..., parents[1:], :], joint_local_quats[..., 1:, :]),
    ), dim=-2)
    return joint_local_quats

def inverse_kinematics_quats(joints_global_rotations: torch.Tensor,joints_global_positions, parents):
    """
    Performs differentiable inverse kinematics with quaternion-based rotations
    :param joints_global_rotations: tensor of global quaternions of shape (..., J, 4)
    :param joints_global_positions: tensor of global positions of shape (..., J, 3)
    :param a list of hierarchy levels, ordered by distance to the root joints in the hierarchy
                           each elements of the list is a list transform indexes
                           for instance level_joints[3] contains a list of indices of all the joints that are 3 levels
                           below the root joints the indices should match these of the provided local rotations
                           for instance if level_joints[4, 1] is equal to 7, it means that local_rotations[:, 7, :, :]
                           contains the local rotation matrix of a joint that is 4 levels deeper than the root joint
                           (ie there are 4 joints above it in the hierarchy)
    :param level_joint_parents: similar to level_transforms but contains the parent indices
                      for instance if level_joint_parents[4, 1] is equal to 5, it means that local_rotations[:, 5, :, :]
                      contains the local rotation matrix of the parent joint of the joint contained in level_joints[4, 1]
    :return:
        joints_local_positions: tensor of local bone offsets of shape (..., J, 3)
        joints_local_rotations: tensor of local quaternions of shape (..., J, 4)
    """


    joint_local_quats = joints_global_rotations.clone()
    invert_global_rotations = quaternion_invert(joint_local_quats)
    joint_local_quats = torch.cat((
        joint_local_quats[..., :1, :],
        quaternion_multiply(invert_global_rotations[..., parents[1:], :], joint_local_quats[..., 1:, :]),
    ), dim=-2)
    joint_local_pos = torch.cat([joints_global_positions[...,:1,:],
                                 quaternion_apply(invert_global_rotations[..., parents[1:], :],joints_global_positions[...,1:,:]-joints_global_positions[...,parents[1:],:])],dim=-2)
    return joint_local_pos,joint_local_quats


from src.geometry.vector import normalize_vector
from src.geometry.quaternions import from_to_quaternion,quat_to_or6D
def scale_pos_to_bonelen(glb_pos:torch.Tensor,edge_len:torch.Tensor,level_joints: List[List[int]], level_joint_parents: List[List[int]]):
    edge_len = edge_len.unsqueeze(1).expand(glb_pos.shape[:2]+(glb_pos.shape[2]-1,3))
    _EPS32 = torch.finfo(torch.float32).eps
    #for level in range(len(level_joints)-1,0,-1):
    # import modifications
    for level in range(1,len(level_joints)):
        parent_bone_indices = level_joint_parents[level]
        local_bone_indices = level_joints[level]
        local_bone_indices_1 = [i-1 for i in local_bone_indices]
        parent_pos = glb_pos[:,:,parent_bone_indices]
        local_pos = glb_pos[:,:,local_bone_indices]
        dir = local_pos - parent_pos
        length = torch.norm(dir,dim=-1,keepdim=True)
        length = torch.where(length<_EPS32,_EPS32,length)
        local_pos = dir/length*edge_len[:,:,local_bone_indices_1]+parent_pos
        glb_pos[:,:,local_bone_indices] = local_pos
    return glb_pos
def change_pos_to_rot(joints_global_rotations: torch.Tensor, joints_global_positions: torch.Tensor,offsets:torch.Tensor,secondary_offsets:torch.Tensor,parent:list):
    """
    Performs differentiable inverse kinematics with quaternion-based rotations
    :param joints_global_rotations: tensor of global quaternions of shape (..., J, 4)
    :param joints_global_positions: tensor of global positions of shape (..., J, 3)
    :param a list of hierarchy levels, ordered by distance to the root joints in the hierarchy
                           each elements of the list is a list transform indexes
                           for instance level_joints[3] contains a list of indices of all the joints that are 3 levels
                           below the root joints the indices should match these of the provided local rotations
                           for instance if level_joints[4, 1] is equal to 7, it means that local_rotations[:, 7, :, :]
                           contains the local rotation matrix of a joint that is 4 levels deeper than the root joint
                           (ie there are 4 joints above it in the hierarchy)
    :param level_joint_parents: similar to level_transforms but contains the parent indices
                      for instance if level_joint_parents[4, 1] is equal to 5, it means that local_rotations[:, 5, :, :]
                      contains the local rotation matrix of the parent joint of the joint contained in level_joints[4, 1]
    :return:
        joints_local_positions: tensor of local bone offsets of shape (..., J, 3)
        joints_local_rotations: tensor of local quaternions of shape (..., J, 4)
    """

    #joint_local_positions = joints_global_positions.clone()
    joint_local_quats = joints_global_rotations.clone()
    joint_offset = joints_global_positions[...,1:,:] - joints_global_positions[...,parent[1:],:]
    joint_offset = normalize_vector(joint_offset)
    offsets = offsets[...,1:,:].unsqueeze(-3).expand(joint_offset.shape)
    offsets = normalize_vector(offsets)
    secondary_offsets = secondary_offsets[...,1:,:].unsqueeze(-3).expand(joint_offset.shape)
    joints_global_rotations = from_to_quaternion(offsets,joint_offset)
    # find plane P of joint offset
    tangent = quaternion_apply(joints_global_rotations, secondary_offsets)
    #co_tangent = normalize_vector(torch.cross(joint_offset,tangent))
    # project the secondary offsets to plane
    init_ore = quaternion_apply(joint_local_quats[...,parent[1:],:],secondary_offsets)

    dot = lambda x,y: (x*y).sum(-1,keepdim=True)
    init_ore = init_ore - dot(init_ore,joint_offset)*joint_offset
    #now init_ore and tangent should be in plane P
    rot = from_to_quaternion(tangent,init_ore)
    joints_global_rotations = quaternion_multiply(rot,joints_global_rotations)

    return joints_global_rotations



