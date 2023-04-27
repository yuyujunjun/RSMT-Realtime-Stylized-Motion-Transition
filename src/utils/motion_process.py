import numpy as np
from np_vector import quat_mul,quat_mul_vec,quat_inv,normalize,quat_between
from BVH_mod import  Anim
def angle_between_quats_along_y(from_quat,to_quat):
    """
        Quats tensor for current rotations (B, 4)
        :return
    """
    #mask = np.tile(np.array([[1.0, 0.0, 1.0]], dtype=np.float),(from_quat.shape[0],1))
    initial_direction = np.array([[0.0, 0.0, 1.0]], dtype=np.float)
    quat = quat_mul(to_quat,quat_inv(from_quat))
    dir = quat_mul_vec(quat,initial_direction)
   # dir[...,1] = 0
   # dir = normalize(dir)
    return np.arctan2(dir[...,0],dir[...,2])
def find_Yrotation_to_align_with_Xplus(q):
    """

    :param q: Quats tensor for current rotations (B, 4)
    :return y_rotation: Quats tensor of rotations to apply to q to align with X+
    """
    mask = np.array([[1.0, 0.0, 1.0]], dtype=np.float)#(q.shape[0], -1)
    mask = np.tile(mask,(q.shape[0],q.shape[1],1))
    '''用来度量quat的方向'''
    initial_direction = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float),(q.shape[0],q.shape[1],1))
    forward = mask * quat_mul_vec(q, initial_direction)
    forward = normalize(forward)
    y_rotation = normalize(quat_between(forward, np.array([[1, 0, 0]])))
    return y_rotation
'''We keep batch version only'''
def extract_feet_contacts(pos:np.array , lfoot_idx, rfoot_idx, velfactor=0.02):
    assert len(pos.shape)==4
    """
    Extracts binary tensors of feet contacts
    :param pos: tensor of global positions of shape (Timesteps, Joints, 3)
    :param lfoot_idx: indices list of left foot joints
    :param rfoot_idx: indices list of right foot joints
    :param velfactor: velocity threshold to consider a joint moving or not
    :return: float tensors of left foot contacts and right foot contacts
    """
    lfoot_xyz = (pos[:,1:, lfoot_idx, :] - pos[:,:-1, lfoot_idx, :]) ** 2
    contacts_l = (np.sum(lfoot_xyz, axis=-1) < velfactor).astype(np.float)

    rfoot_xyz = (pos[:,1:, rfoot_idx, :] - pos[:,:-1, rfoot_idx, :]) ** 2
    contacts_r = ((np.sum(rfoot_xyz, axis=-1)) < velfactor).astype(np.float)

    # Duplicate the last frame for shape consistency
    contacts_l = np.concatenate([contacts_l, contacts_l[:,-1:]], axis=1)
    contacts_r = np.concatenate([contacts_r, contacts_r[:,-1:]], axis=1)

    return [contacts_l,contacts_r]

def subsample(anim:Anim,ratio = 2):
    anim.quats = anim.quats[::ratio,...]
    anim.hip_pos = anim.hip_pos[::ratio,...]
    return anim