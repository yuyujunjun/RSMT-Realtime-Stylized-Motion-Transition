import random

import numpy as np
import pytorch3d
import torch.utils.data
from pytorch3d.transforms import quaternion_apply, quaternion_multiply

from src.Datasets.BaseLoader import BasedDataProcessor
from src.Datasets.augmentation import angle_between_quats_along_y
from src.geometry.quaternions import quat_to_or6D, or6d_to_quat, from_to_1_0_0


class BatchProcessDatav2(torch.nn.Module):
    def __init__(self):
        super(BatchProcessDatav2, self).__init__()
    def forward(self,glb_rot,glb_pos):
        dir = glb_pos[..., 5:6, :] - glb_pos[..., 1:2, :]
        ref_vector = torch.cat((-dir[...,2:3],torch.zeros_like(dir[...,1:2]),dir[...,0:1]),dim=-1)
        root_rotation = from_to_1_0_0(ref_vector)
        glb_vel = quaternion_apply(root_rotation[:,:-1], glb_pos[:,1:]-glb_pos[:,:-1])
        glb_rot = quaternion_multiply(root_rotation,glb_rot)
        glb_pos = glb_pos.clone()
        glb_pos[...,[0,2]] = glb_pos[...,[0,2]]-glb_pos[...,0:1,[0,2]]
        glb_pos = quaternion_apply(root_rotation,glb_pos)
        return glb_vel,glb_pos,glb_rot,root_rotation
class BatchUnProcessDatav2(torch.nn.Module):
    # we don't need use it in training
    def __init__(self):
        super(BatchUnProcessDatav2, self).__init__()
    def forward(self,glb_pos,glb_rot,glb_vel,root_rotation):
        # glb_pos: N,T,J,3
        inverse_rot = pytorch3d.transforms.quaternion_invert(root_rotation)
        glb_pos = quaternion_apply(inverse_rot,glb_pos)
        out_pos = torch.empty(size=glb_rot.shape[:-1]+(3,),dtype=glb_pos.dtype,device=glb_pos.device)
        out_pos[:,0:1,:,:] = glb_pos[:,0:1,:,:]

        glb_vel = quaternion_apply(inverse_rot[:,:-1],glb_vel)
        for i in range(0,glb_vel.shape[1]):
            out_pos[:,i+1,...] = out_pos[:,i]+glb_vel[:,i]
        glb_rot = quaternion_multiply(inverse_rot,glb_rot)
        #glb_pos = out_pos
        glb_pos[...,[0,2]]=out_pos[:,:,0:1,[0,2]]+glb_pos[...,[0,2]]
        return  glb_pos,glb_rot



class BatchProcessData(torch.nn.Module):#(B,T,J,dim)
    def __init__(self):
        super(BatchProcessData, self).__init__()
    def forward(self,global_rotations,global_positions):
        ref_vector = torch.cross(global_positions[...,5:6,:]-global_positions[...,1:2,:],torch.tensor([0,1,0],dtype=global_positions.dtype,device=global_positions.device),dim=-1)

        root_rotation = from_to_1_0_0(ref_vector)
        """ Local Space """
        local_positions = global_positions.clone()
        local_positions[..., 0] = local_positions[..., 0] - local_positions[..., 0:1, 0]
        local_positions[...,2] = local_positions[..., 2] - local_positions[..., 0:1, 2]

        local_positions = quaternion_apply(root_rotation, local_positions)
        local_velocities = quaternion_apply(root_rotation[:,:-1], (global_positions[:,1:] - global_positions[:,:-1]))
        local_rotations = quaternion_multiply(root_rotation, global_rotations)

        local_rotations = quat_to_or6D(local_rotations)
        if torch.isnan(local_rotations).any():
            print("6D")
            assert False
        root_global_velocity_XZ = global_positions[:,1:, 0:1] - global_positions[:,:-1, 0:1]
        root_global_velocity_XZ[..., 1] = 0
        root_velocity = quaternion_apply(root_rotation[:,:-1, :], (root_global_velocity_XZ))
        #root_rvelocity = angle_between_quats_along_y(global_rotations[:,:-1, 0:1, :], global_rotations[:,1:, 0:1, :])
        root_rvelocity = angle_between_quats_along_y(root_rotation[:,1:, 0:1, :], root_rotation[:,:-1, 0:1, :])
        dict = {"root_rotation":root_rotation,
            "local_pos":local_positions[:,:,...],"local_vel":local_velocities,"local_rot":local_rotations[:,:,...],"root_vel":root_velocity[...,[0,2]],"root_rvel":root_rvelocity.unsqueeze(-1)}

        return dict
class UnProcessData(torch.nn.Module):
    def __init__(self):
        super(UnProcessData, self).__init__()

    def get_global_rotation(self,root_rvelocity):
        '''root_rvelocity:  N,T,...,C'''
        #r = torch.zeros_like(root_rvelocity)  # BxTx1 or Bx1 or BxTx1x1
        shape = list(root_rvelocity.shape)
        shape[1]+=1
        r = torch.zeros(shape,device=root_rvelocity.device)
        for i in range(1, r.shape[1]):
                r[:, i] = r[:, i - 1] + root_rvelocity[:, i - 1]
        shape = [1 for i in range(r.dim())]
        shape[-1] = 3
        axis = torch.zeros(size = shape,device=r.device)
        axis[...,0:3] = torch.tensor([0,1,0])
        rotation = pytorch3d.transforms.axis_angle_to_quaternion(axis * r)  # B,T,1,4
        return rotation
    def get_global_xz(self,global_hip_rot,hip_velocity):
        root_velocity3D = torch.zeros(size=(hip_velocity.shape[0], hip_velocity.shape[1], 1, 3),device=global_hip_rot.device,dtype=hip_velocity.dtype)
        root_velocity3D[..., [0, 2]] = hip_velocity.clone()
        root_velocity3D = quaternion_apply(global_hip_rot[:,:-1,...], root_velocity3D)
        shape = list(root_velocity3D.shape)
        shape[1]+=1
        root_pos_XZ = torch.zeros(shape,device=root_velocity3D.device)
        for i in range(1, (root_velocity3D.shape[1]+1)):
            root_pos_XZ[:, i, 0, :] = root_pos_XZ[:, i - 1, 0, :] + root_velocity3D[:, i - 1, 0, :]
        return root_pos_XZ



    def forward(self,batch):
        num_joints = batch['local_pos'].shape[-2]
        batch_size = batch['local_pos'].shape[0]
        root_rvelocity = batch['root_rvel']
        local_positions = batch['local_pos']
        local_rotations = batch['local_rot']
        local_rotations = or6d_to_quat(local_rotations)
        root_velocity = batch['root_vel']



        rotation = self.get_global_rotation(root_rvelocity)
        '''transform to global rotation'''
        global_rot = quaternion_multiply(rotation,local_rotations)
        relative_pos = quaternion_apply(rotation,local_positions)
        global_pos_xz = self.get_global_xz(rotation,root_velocity)
        '''transform to global pos'''
        global_pos = relative_pos + global_pos_xz
        #return relative_pos,local_rotations
       # global_pos = local_positions
      #  global_rot = local_rotations
        batch['global_pos'] = global_pos
        batch['global_rot'] = global_rot
        return global_pos,global_rot

class BatchRotateYCenterXZ(torch.nn.Module):
    def __init__(self):
        super(BatchRotateYCenterXZ, self).__init__()
    def forward(self,global_positions,global_quats,ref_frame_id):
        ref_vector = torch.cross(global_positions[:, ref_frame_id:ref_frame_id+1, 5:6, :] - global_positions[:, ref_frame_id:ref_frame_id+1, 1:2, :],
                                 torch.tensor([0, 1, 0], dtype=global_positions.dtype, device=global_positions.device).view(1,1,1,3),dim=-1)
        root_rotation = from_to_1_0_0(ref_vector)

        ref_hip = torch.mean(global_positions[:,:,0:1,[0,2]],dim=(1),keepdim=True)
        global_positions[...,[0,2]] = global_positions[...,[0,2]] - ref_hip
        global_positions = quaternion_apply(root_rotation, global_positions)
        global_quats = quaternion_multiply(root_rotation, global_quats)
        """ Local Space """


        return global_positions,global_quats

