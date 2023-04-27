from typing import List

import torch
from pytorch3d.transforms import quaternion_multiply, quaternion_apply

def forward_transform_hierarchy(local_transforms: torch.Tensor, level_transforms: List[List[int]],
                                level_transform_parents: List[List[int]]):
    # used to store world transforms
    world_transforms = torch.zeros_like(local_transforms)

    # initialize root transforms
    world_transforms[:, level_transforms[0]] = local_transforms[:, level_transforms[0], :, :]

    # then process all children transforms
    for level in range(1, len(level_transforms)):
        parent_bone_indices = level_transform_parents[level]
        local_bone_indices = level_transforms[level]
        parent_level_transforms = world_transforms[:, parent_bone_indices]
        local_level_transforms = local_transforms[:, local_bone_indices]
        global_matrix = torch.matmul(parent_level_transforms, local_level_transforms)
        world_transforms[:, local_bone_indices] = global_matrix.type_as(world_transforms)

    return world_transforms


# Compute a transform matrix combining a rotation and translation (rotation is applied first)
# translations: translation vectors (N, 3)
# rotations: rotation matrices (N, 3, 3)
# out: transform matrix (N, 4, 4)
def get_transform_matrix(translations: torch.Tensor, rotations: torch.Tensor):
    batch_size = rotations.shape[0]

    out = torch.zeros((batch_size, 4, 4)).type_as(rotations)
    out[:, 0:3, 0:3] = rotations
    out[:, 0:3, 3] = translations
    out[:, 3, 3] = 1.0

    return out




def forward_kinematics(local_rotations: torch.Tensor, local_offsets: torch.Tensor,
                       level_joints: List[List[int]], level_joint_parents: List[List[int]]):
    batch_size = local_rotations.shape[0]
    joints_nbr = local_rotations.shape[1]

    # compute local transformation matrix for each joints
    local_transforms = get_transform_matrix(translations=local_offsets.view(-1, 3), rotations=local_rotations.view(-1, 3, 3))
    local_transforms = local_transforms.view(batch_size, joints_nbr, 4, 4)

    return forward_transform_hierarchy(local_transforms, level_joints, level_joint_parents)
def forward_kinematics_quats(local_rotations: torch.Tensor, local_offsets: torch.Tensor, hip_positions:torch.Tensor,
                       parents):
    """
    Performs FK to retrieve global positions and rotations in quaternion format
    :param local_rotations: tensor of local rotations of shape (..., J, 4)
    :param local_offsets: tensor of local bone offsets of shape (..., J, 3)
    :param level_joints:  a list of hierarchy levels, ordered by distance to the root joints in the hierarchy
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
        global_positions : tensor of global bone positions of shape (..., J, 3)
        global_quats     : tensor of global bone quaternions of shape (..., J, 4)
    """

    quat_shape = list(local_rotations.shape)
    pos_shape = quat_shape.copy()
    pos_shape[-1] = 3
    global_quats = local_rotations.clone()
    local_offsets = local_offsets.unsqueeze(-3).expand(pos_shape)
    global_positions = [hip_positions]
    global_quats = [global_quats[...,:1,:]]
    #global_positions[...,0:1,:] = hip_positions.clone()
    for level in range(1, len(parents)):

        global_positions.append( quaternion_apply(global_quats[parents[level]], local_offsets[..., level:level+1, :]) + global_positions[parents[level]])
        global_quats.append( quaternion_multiply(global_quats[parents[level]], local_rotations[..., level:level+1, :]))
    global_positions = torch.cat(global_positions,dim=-2)
    global_quats = torch.cat(global_quats,dim=-2)
    return global_positions, global_quats

