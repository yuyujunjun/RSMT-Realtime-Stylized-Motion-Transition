import re
import numpy as np
import src.geometry.forward_kinematics as fk
import src.geometry.inverse_kinematics as ik
import torch
from src.geometry.vector import find_secondary_axis
from np_vector import euler_to_quat,remove_quat_discontinuities,quat_to_euler
import pytorch3d.transforms as trans


channelmap = {
    'Xrotation' : 'x',
    'Yrotation' : 'y',
    'Zrotation' : 'z'   
}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x' : 0,
    'y' : 1,
    'z' : 2,
}
def find_sys_with_prefix(names,name,l_prefix,r_prefix):
    if (name.lower().startswith(l_prefix)):
        l_joint_name = name[len(l_prefix):]
        for rName in names:
            if(rName.lower().startswith(r_prefix)):
                r_joint_name = rName[len(r_prefix):]
                if(l_joint_name == r_joint_name):
                    return rName
    return ""
def find_sys(names,name):
    sys_part = find_sys_with_prefix(names,name,"l","r")
    if(sys_part==""):
        sys_part = find_sys_with_prefix(names, name, "left", "right")
    return sys_part
def find_level_recursive(map,parent,node,level_joint,level_joint_parent,level):
    if(level not in level_joint):
        level_joint[level]=[]
    if(level not in level_joint_parent):
        level_joint_parent[level]=[]
    level_joint[level].append(node)
    level_joint_parent[level].append(parent)
    for child_id in map[node]["children"]:
        find_level_recursive(map,node,child_id,level_joint,level_joint_parent,level+1)
def find_link_recursive(map,node,link):
    for child_id in map[node]["children"]:
        link.append((node,child_id))
        find_link_recursive(map,child_id,link)
class Skeleton(object):
    def __init__(self,parents,names,offsets,symmetry=None):
        self.num_joints = len(parents)
        self.root = 0
        self.parents = parents
        self.names = names
        self.hierarchy = {}
        self.name_to_id = {}
        self.__build_name_map()
        self.bone_pair_indices = symmetry
        self._level_joints = []
        self._level_joints_parents=[]
        if(symmetry==None):
            self.__predict_sysmetry()
        self.predict_sysmetry_axis(offsets)
        self.__build_hierarchy()
        self.__build_level_map()
        self.self_link = [(i,i) for i in range(len(parents))]
        self.neighbor_link =[]
        find_link_recursive(self.hierarchy,self.root,self.neighbor_link)
        self.build_end_effector()


    def __eq__(self, other):
        if(other==None):
            return False
        return self.parents == other.parents and self.names == other.names and self.bone_pair_indices == other.bone_pair_indices
    def build_end_effector(self):
        self.multi_children_idx = []
        self.end_effector_idx = []
        for i in range(len(self.names)):
            child = [*self.hierarchy[i]['children'].keys()]
            if(len(child)>1):
                self.multi_children_idx.append(i)
            elif(len(child)==0):
                self.end_effector_idx.append(i)

    def __build_name_map(self):
        self.hierarchy[-1] = {"name":""}
        self.name_to_id[''] = -1
        for idx, name in enumerate(self.names):
            self.hierarchy[idx] = {"name": name}
            self.name_to_id[name] = idx
    def predict_sysmetry_axis(self,offsets):
        offsets = offsets[self.bone_pair_indices]-offsets
        x = np.max(np.abs(offsets)[...,0])
        y = np.max(np.abs(offsets)[...,1])
        z = np.max(np.abs(offsets)[...,2])
        if(x>y and x>z):
            self.sys = "x"
        elif(y>z and y>x):
            self.sys = 'y'
        elif(z>y and z>x):
            self.sys = 'z'
        else:
            assert(0,"Unpredicition")
    def __predict_sysmetry(self):
        '''根据骨骼名字猜测对称骨骼idx，可能不准确，如果没有对称骨骼，则该位置为-1'''
        self.bone_pair_indices = [-1 for _ in range(len(self.parents))]
        for idx, name in enumerate(self.names):
            if(idx in self.bone_pair_indices):
                continue
            sys_name = (find_sys(self.names,name))
            sys_idx = self.name_to_id[sys_name]
            self.bone_pair_indices[idx] = sys_idx
            if(sys_idx==-1):
                self.bone_pair_indices[idx] = idx
            else:
                self.bone_pair_indices[sys_idx] = idx



    def __build_hierarchy(self):
        '''构建骨骼名字，idx，parent,children，对称等索引'''
        mapping = self.hierarchy
        for idx, (name,par_id) in enumerate(zip(self.names, self.parents)):
            children_id = []
            children = {}
            for child_id,child_parent in enumerate(self.parents):
                if(child_parent==idx):
                    children_id.append(child_id)
                    children[child_id] = mapping[child_id]
            mapping[idx]["parent"] = mapping[par_id]
            mapping[idx]["children"] = children
            if(self.bone_pair_indices[idx]!=-1):
                mapping[idx]["sys"] = mapping[self.bone_pair_indices[idx]]

    def __build_level_map(self):
        level_joint = {}
        level_joint_parent = {}

        find_level_recursive(self.hierarchy,-1,self.root,level_joint,level_joint_parent,0)
        for i in range(len(level_joint)):
            self._level_joints.append(level_joint[i])
            self._level_joints_parents.append(level_joint_parent[i])

    def forward_kinematics(self,local_rotations: torch.Tensor, local_offsets: torch.Tensor,hip_positions:torch.Tensor):
        #gp,gq =  fk.forward_kinematics_quats(local_rotations, local_offsets,hip_positions, self._level_joints, self._level_joints_parents)
        gp,gq =  fk.forward_kinematics_quats(local_rotations, local_offsets,hip_positions, self.parents)
        return gp,gq
    def inverse_kinematics_quats(self,joints_global_rotation:torch.Tensor):
        return ik.inverse_kinematics_only_quats(joints_global_rotation, self.parents)
    def inverse_kinematics(self,joints_global_rotations: torch.Tensor, joints_global_positions: torch.Tensor,output_other_dim = False):
        lp,lq =  ik.inverse_kinematics_quats(joints_global_rotations, joints_global_positions, self.parents)
        if(output_other_dim):
            return lp,lq
        else:
            return lp[...,0:1,:],lq
    def global_rot_to_global_pos(self,global_rot,offsets,hip):
        # quick version of fk(ik()), only for global_rotation
        pos_shape = global_rot.shape[:-1]+(3,)
        offsets = offsets.unsqueeze(-3).expand(pos_shape)
        #hip = torch.zeros_like(offsets[...,:1,:])
        gp = torch.empty(pos_shape,dtype=global_rot.dtype,device = global_rot.device)#[hip]
        gp[...,0:1,:] = hip[...,0:1,:]
        offset = trans.quaternion_apply(global_rot[..., self.parents[1:], :], offsets[..., 1:, :])

        for level in range(1, len(self._level_joints)):
            local_bone_indices = self._level_joints[level]
            local_bone_indices_1 = [i-1 for i in local_bone_indices]
            parent_bone_indices = self._level_joints_parents[level]
            gp[...,local_bone_indices,:] = offset[...,local_bone_indices_1,:]+gp[...,parent_bone_indices,:]

        # for i in range(1, len(self.parents)):
        #     gp.append(offset[..., i-1:i, :] + gp[self.parents[i]])
        # gp = torch.cat(gp, dim=-2)
        return gp
    def inverse_pos_to_rot(self,joints_global_rotations,joints_global_positions,offsets,tangent):
        edge_len = torch.norm(offsets[:,1:],dim=-1,keepdim=True)
        joints_global_positions = ik.scale_pos_to_bonelen(joints_global_positions,edge_len,self._level_joints,self._level_joints_parents)
        parent_rot = self.pos_to_parent_rot(joints_global_rotations,joints_global_positions,offsets,tangent)
        out = self.reduce_parent_rot(parent_rot)
        special_idx = [*self.end_effector_idx,*self.multi_children_idx,*[9,10,11,12,13,15,19]]
        out[...,special_idx,:] = joints_global_rotations[...,special_idx,:]
        return out

    def pos_to_parent_rot(self,joints_global_rotations:torch.Tensor,joints_global_positions:torch.Tensor,local_offsets:torch.Tensor,secondary_offsets:torch.Tensor):
        return ik.change_pos_to_rot(joints_global_rotations, joints_global_positions, local_offsets, secondary_offsets,self.parents)
    def reduce_parent_rot(self,parent_rot):
        shape = list(parent_rot.shape)
        shape[-2] += 1
        rot = torch.empty(size=shape,device=parent_rot.device,dtype=parent_rot.dtype)
        #children = [[*self.hierarchy[i]['children'].keys()] for i in range(len(self.names))]
        idx = []
        child_idx = []
        for i in range(len(self.names)):
            child = [*self.hierarchy[i]['children'].keys()]
            if(len(child)==1):
                idx.append(i)
                child_idx.append(child[0]-1)
        rot[...,idx,:] = parent_rot[...,child_idx,:]
            # else:
            #     rot[..., i, :] = torch.zeros_like(rot[..., i, :])
        return rot



class SkeletonLevel():
    def __init__(self,n_joints,edges,root,body_parts,last_n_joints, pooling_list=None,unpooling_list=None):
        self.n_joints = n_joints
        self.last_n_joints = last_n_joints
        self.edges = edges
        self.pooling_list = pooling_list
        self.unpooling_list = unpooling_list
        self.root = root
        self.body_parts = body_parts







class Anim(object):

    """
    A very basic animation object
    """
    def __init__(self, quats, pos, offsets, parents, names,remove_joints = None,Tpose = -1,remove_gap:float = 0.,filename= ""):
        """
        :param quats: local quaternions tensor
        :param pos: local positions tensor
        :param offsets: local joint offsets
        :param parents: bone hierarchy
        :param names: bone names
        :param valid_names: useful for remove unnecessary joints
        """
        from src.Datasets.BatchProcessor import BatchProcessData
        process = BatchProcessData()
        self.quats = quats
        self.hip_pos = pos
        self.offsets = offsets

        self.parents = parents
        self.names = names
        if(remove_joints!=None):
            self.quats,self.offsets,self.parents,self.names = remove_joints(self.quats,self.offsets,self.parents,self.names)

        self.skeleton = Skeleton(self.parents,self.names,self.offsets)
        if(remove_gap>0.):

            gp,gq = self.skeleton.forward_kinematics(torch.from_numpy(self.quats),torch.from_numpy(self.offsets),torch.from_numpy(self.hip_pos))

            idx = check_continuty(gp,remove_gap)
            #print(filename)

            idx = list(set(idx))

            idx.sort()
            dict = process(gq.unsqueeze(0),gp.unsqueeze(0))
            lp = dict['local_pos'][0]
            idx1 = check_continuty(lp,remove_gap)
            idx1 = list(set(idx1))
            idx1.sort()
            if(idx[-1]>10 or idx1[-1]>10):
                print(filename)
                print()
            idx = max(idx[-1],idx1[-1])+1
            self.quats = self.quats[idx:]
            self.hip_pos = self.hip_pos[idx:]
        self.secondary_offsets = find_secondary_axis(torch.from_numpy(self.offsets)).numpy()
    def forward_kinamatics(self):
        return self.skeleton.forward_kinematics(torch.from_numpy(self.quats),torch.from_numpy(self.offsets),torch.from_numpy(self.hip_pos))







def check_continuty(x,threshold):
    nei = (x[1:]-x[:-1]).abs()
    index = np.where(nei>threshold)[0]
    index = (set(index))

    vel = (nei[1:]-nei[:-1]).abs()
    id = np.where(vel>threshold/10)[0]
    id = (set(id))

    index = index.intersection(id)
    index = list(index)
    index.sort()
    if (len(index) == 0):
        return [-1]
    seq_start = index[-1]+1
    if(seq_start>10):
        print({key:float(nei[key].max()) for key in index})

       # print("remove gap:{}, max gap{}, remove gap{}".format(seq_start,nei.max(),nei[seq_start-1].max()))

    return index
def remove_zero_seq(hip_pos):
    '''把一段序列中hip_pos在原点之前的子序列移除'''
    '''[seq,1,3]'''
    index = np.where(np.dot(np.abs(hip_pos),np.array([1,1,1]))==0)
    if(index[0].shape[0]==0):
        return 0
    seq_start = index[0][-1]
    return seq_start

def read_bvh(filename, start=None, end=None, order=None,remove_joints = None,Tpose = -1,remove_gap :float=0.):
    print(filename)
    """
    Reads a BVH file and extracts animation information.

    :param filename: BVh filename
    :param start: start frame
    :param end: end frame
    :param order: order of euler rotations
    :return: A simple Anim object conatining the extracted information.
    """

    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False

    names = []
    orients = np.array([]).reshape((0, 4))
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)

    # Parse the  file, line by line
    for line in f:

        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        rmatch = re.match(r"ROOT (\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "{" in line: continue

        if "}" in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue

        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            if not end_site:
                offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis:2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
            continue

        jmatch = re.match("\s*JOINT\s+(\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "End Site" in line:
            end_site = True
            continue

        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            if start and end:
                fnum = (end - start) - 1
            else:
                fnum = int(fmatch.group(1))
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        if (start and end) and (i < start or i >= end - 1):
            i += 1
            continue

        dmatch = line.strip().split(' ')
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents)
            fi = i - start if start else i
            if channels == 3:
                positions[fi, 0:1] = data_block[0:3]
                rotations[fi, :] = data_block[3:].reshape(N, 3)
            elif channels == 6:
                data_block = data_block.reshape(N, 6)
                positions[fi, :] = data_block[:, 0:3]
                rotations[fi, :] = data_block[:, 3:6]
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(N - 1, 9)
                rotations[fi, 1:] = data_block[:, 3:6]
                positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1

    f.close()

    rotations = euler_to_quat(np.radians(rotations), order=order)
    rotations = remove_quat_discontinuities(torch.from_numpy(rotations).unsqueeze(0))[0].numpy()

    return Anim(rotations, positions[...,0:1,:], offsets, parents, names,remove_joints,Tpose=Tpose,remove_gap=remove_gap,filename=filename)
    

    
#anim.pos: only root position writen into file, anim.offset: write all joints to file
def save_bvh(filename, anim, names=None, frametime=1.0 / 30.0, order='xyz', positions=False, orients=True):
    """
    Saves an Animation to file as BVH

    Parameters
    ----------
    filename: str
        File to be saved to

    anim : Animation
        Animation to save

    names : [str]
        List of joint names

    order : str
        Optional Specifier for joint order.
        Given as string E.G 'xyz', 'zxy'

    frametime : float
        Optional Animation Frame time

    positions : bool
        Optional specfier to save bone
        positions for each frame

    orients : bool
        Multiply joint orients to the rotations
        before saving.

    """

    if names is None:
        if anim.names is None:
            names = ["joint_" + str(i) for i in range(len(anim.parents))]
        else:
            names = anim.names
    print(filename)
    with open(filename, 'w') as f:

        t = ""
        f.write("%sHIERARCHY\n" % t)
        f.write("%sROOT %s\n" % (t, names[0]))
        f.write("%s{\n" % t)
        t += '\t'

        f.write("%sOFFSET %f %f %f\n" % (t, anim.offsets[0, 0], anim.offsets[0, 1], anim.offsets[0, 2]))
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" %
                (t, channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))
        num_joints=anim.parents.shape[0]
        assert num_joints==anim.offsets.shape[0]
        seq_length=anim.quats.shape[0]
        assert seq_length==anim.hip_pos.shape[0]
        for i in range(anim.parents.shape[0]):#iterate joints
            if anim.parents[i] == 0:
                t = save_joint(f, anim, names, t, i, order=order, positions=positions)

        t = t[:-1]
        f.write("%s}\n" % t)

        f.write("MOTION\n")
        f.write("Frames: %i\n" % anim.quats.shape[0]);
        f.write("Frame Time: %f\n" % frametime);


        # if orients:
        #    rots = np.degrees((-anim.orients[np.newaxis] * anim.rotations).euler(order=order[::-1]))
        # else:
        #    rots = np.degrees(anim.rotations.euler(order=order[::-1]))
        rots = np.degrees(quat_to_euler(anim.quats))
        poss = anim.hip_pos
        if len(poss.shape)==2:
            poss=poss[:,np.newaxis,:]
        for i in range(poss.shape[0]):#frame
            for j in range(anim.parents.shape[0]):#joints

                if positions or j == 0:

                    f.write("%f %f %f %f %f %f " % (
                        poss[i, j, 0], poss[i, j, 1], poss[i, j, 2],
                        rots[i, j, ordermap[order[0]]], rots[i, j, ordermap[order[1]]], rots[i, j, ordermap[order[2]]]))

                else:

                    f.write("%f %f %f " % (
                        rots[i, j, ordermap[order[0]]], rots[i, j, ordermap[order[1]]], rots[i, j, ordermap[order[2]]]))

            f.write("\n")


def save_joint(f, anim, names, t, i, order='zyx', positions=False):
    f.write("%sJOINT %s\n" % (t, names[i]))
    f.write("%s{\n" % t)
    t += '\t'

    f.write("%sOFFSET %f %f %f\n" % (t, anim.offsets[i, 0], anim.offsets[i, 1], anim.offsets[i, 2]))

    if positions:
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % (t,
                                                                            channelmap_inv[order[0]],
                                                                            channelmap_inv[order[1]],
                                                                            channelmap_inv[order[2]]))
    else:
        f.write("%sCHANNELS 3 %s %s %s\n" % (t,
                                             channelmap_inv[order[0]], channelmap_inv[order[1]],
                                             channelmap_inv[order[2]]))

    end_site = True

    for j in range(anim.parents.shape[0]):
        if anim.parents[j] == i:
            t = save_joint(f, anim, names, t, j, order=order, positions=positions)
            end_site = False

    if end_site:
        f.write("%sEnd Site\n" % t)
        f.write("%s{\n" % t)
        t += '\t'
        f.write("%sOFFSET %f %f %f\n" % (t, 0.0, 0.0, 0.0))
        t = t[:-1]
        f.write("%s}\n" % t)

    t = t[:-1]
    f.write("%s}\n" % t)

    return t
