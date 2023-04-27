import random
from typing import Optional

import numpy as np
import torch
from pytorch3d.transforms import quaternion_multiply, quaternion_apply, quaternion_invert
from src.geometry.quaternions import slerp, from_to_1_0_0
from src.geometry.vector import normalize_vector
from src.utils.BVH_mod import Skeleton
def find_Yrotation_to_align_with_Xplus(q):
    '''Warning: the quat must be along y axis'''
    """
    :param q: Quats tensor for current rotations (B, 4)
    :return y_rotation: Quats tensor of rotations to apply to q to align with X+
    """
    shape = list(q.shape)
    q = q.reshape(-1,4)
    mask = torch.tensor(np.array([[1.0, 0.0, 1.0]]), dtype=q.dtype,device=q.device).expand(q.shape[0], -1)
    ref_vector = torch.tensor(np.array([[1.0, 0.0, 0.0]]), dtype=q.dtype,device=q.device).expand(q.shape[0], -1)
    forward = mask * quaternion_apply(q, ref_vector)
    #forward = normalize_vector(forward)
    #target =  torch.tensor(np.array([[1, 0, 0]]), dtype=torch.float,device=q.device).expand(q.shape[0], -1)

    #y_rotation = normalize_vector(from_to_quaternion(forward, target))
    y_rotation = normalize_vector(from_to_1_0_0(forward))



    return y_rotation.view(shape)
def angle_between_quats_along_y(from_quat:torch.Tensor,to_quat:torch.Tensor):
    """
        Quats tensor for current rotations (B, 4)
        :return
    """

    #mask = np.tile(np.array([[1.0, 0.0, 1.0]], dtype=np.float),(from_quat.shape[0],1))
    initial_direction = torch.tensor(np.array([[0.0, 0.0, 1.0]]), dtype=torch.float,device=from_quat.device).expand(from_quat.shape[:-1]+(3,))
    quat = quaternion_multiply(to_quat,quaternion_invert(from_quat))
    dir = quaternion_apply(quat,initial_direction)

    return torch.atan2(dir[...,0],dir[...,2])





class TemporalScale(torch.nn.Module):
    def __init__(self,prob):
        super(TemporalScale, self).__init__()
        self.prob = prob
    def forward(self,hip_pos,quat):
        if (random.random() > self.prob):
            return hip_pos, quat
            # hip_pos = batch['hip_pos']
            # quat = batch['quat']
        batch_size = hip_pos.shape[0]
        T = hip_pos.shape[1]
        num_joints = quat.shape[2]
        delta_t = random.randint(T // 2, T)
        start_t = random.randint(0, T - delta_t)
        hip_pos = hip_pos[:, start_t:delta_t + start_t, :, :]
        quat = quat[:, start_t:delta_t + start_t, :, :]
        if delta_t < 0.75 * T:
            gamma = random.random() + 1  # gamma \in [1,2]
        else:
            gamma = random.random() * 0.5 + 0.5  # gamma \in [0.5,1]
        new_delta_t = int(delta_t * gamma)
        new_hip_pos = torch.zeros((batch_size, new_delta_t, 1, 3), device=hip_pos.device, dtype=hip_pos.dtype)
        new_quat = torch.zeros((batch_size, new_delta_t, num_joints, 4), device=hip_pos.device, dtype=hip_pos.dtype)
        '''scale'''
        ratio = delta_t / new_delta_t
        idx = torch.arange(0, new_delta_t, device=quat.device).type(torch.long)
        ori_frame_step = idx * ratio
        start_frame = ori_frame_step.to(torch.long)
        neighbor_frame = start_frame + 1
        neighbor_frame = torch.where(neighbor_frame >= quat.shape[1], neighbor_frame - 1, neighbor_frame)
        weight = start_frame - ori_frame_step + 1
        weight = weight.view(1, -1, 1, 1)

        new_quat[:, idx, :, :] = slerp(quat[:, start_frame, :, :], quat[:, neighbor_frame, :, :], 1 - weight)
        new_hip_pos[:, idx, :, :] = hip_pos[:, start_frame, :, :] * weight + (1 - weight) * hip_pos[:, neighbor_frame,:, :]

        '''padding or cutting'''
        if new_delta_t > T:
            new_quat = new_quat[:, :T, :, :]
            new_hip_pos = new_hip_pos[:, :T, :, :]
        # elif new_delta_t < self.T:
        #     new_quat = pad_to_window(new_quat, window=self.T)
        #     new_hip_pos = pad_to_window(new_hip_pos, window=self.T)

        return new_hip_pos, new_quat




class BatchMirror(torch.nn.Module):
    def __init__(self, skeleton, mirror_prob=0.5):
        super().__init__()
        self.mirroring = Mirror(skeleton=skeleton,dtype=torch.float)
        self.mirror_probability = mirror_prob
        self.nb_joints = skeleton.num_joints

    def forward(self, global_pos,global_rot):
        if torch.rand(1)[0] < self.mirror_probability:
            batch_size = global_pos.shape[0]
            original_positions = global_pos.reshape(-1, self.nb_joints, 3)
            original_rotations = global_rot.reshape(-1, self.nb_joints, 4)
            mirrored_positions, mirrored_rotations = self.mirroring.forward(original_positions, original_rotations)
            global_pos = mirrored_positions.view(batch_size, -1, self.nb_joints, 3)
            global_rot = mirrored_rotations.view(batch_size, -1, self.nb_joints, 4)
        return global_pos,global_rot



class Mirror(torch.nn.Module):
    def __init__(self, skeleton: Skeleton,dtype:torch.dtype):
        super().__init__()
        self.register_buffer('reflection_matrix', torch.eye(3,dtype=dtype))
        # quaternions must be in w, x, y, z order
        if(skeleton.sys=="x"):# x
            self.reflection_matrix[0, 0] = -1
            self.register_buffer('quat_indices', torch.tensor([2, 3]))
        elif(skeleton.sys=='z'):# z
            self.reflection_matrix[2, 2] = -1
            self.register_buffer('quat_indices', torch.tensor([1, 2]))
        elif(skeleton.sys=='y'):
            self.reflection_matrix[1,1] = -1
            self.register_buffer('quat_indices', torch.tensor([1, 3]))
        else:
            assert(0,"can't mirror")




        self.register_buffer('swap_index_list', torch.LongTensor(skeleton.bone_pair_indices))

    def forward(self, joint_positions: torch.Tensor = None, joint_rotations: torch.Tensor = None):#t position in global space
        self.reflection_matrix = self.reflection_matrix.to(joint_positions.device)
        if joint_positions is None:
            new_positions = None
        else:
            new_positions = self._swap_tensor(joint_positions)
            new_positions = torch.matmul(self.reflection_matrix, new_positions.permute(1, 2, 0)).permute(2, 0, 1)
            new_positions = new_positions.contiguous()

        if joint_rotations is None:
            new_rotations = None
        else:
            new_rotations = self._swap_tensor(joint_rotations)
            new_rotations[:, :, self.quat_indices] *= -1
            new_rotations = new_rotations.contiguous()

        return new_positions, new_rotations

    def _swap_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, self.swap_index_list, :].view_as(tensor)


