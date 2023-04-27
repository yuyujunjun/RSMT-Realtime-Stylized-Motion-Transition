import torch
from pytorch3d.transforms import quaternion_apply, quaternion_multiply, quaternion_invert
from src.Datasets.Style100Processor import StyleLoader
from src.geometry.quaternions import or6d_to_quat, quat_to_or6D, from_to_1_0_0
from src.utils import BVH_mod as BVH
from src.utils.BVH_mod import Skeleton, find_secondary_axis


def load_model():
    model = torch.load('./results/Transitionv2_style100/myResults/141/m_save_model_198')
    return model
def load_dataSet():
    loader = StyleLoader()
    loader.load_dataset("+phase_gv10")
    return loader
class BatchRotateYCenterXZ(torch.nn.Module):
    def __init__(self):
        super(BatchRotateYCenterXZ, self).__init__()
    def forward(self,global_positions,global_quats,ref_frame_id):
        ref_vector = torch.cross(global_positions[:, ref_frame_id:ref_frame_id+1, 5:6, :] - global_positions[:, ref_frame_id:ref_frame_id+1, 1:2, :],
                                 torch.tensor([0, 1, 0], dtype=global_positions.dtype, device=global_positions.device),dim=-1)
        root_rotation = from_to_1_0_0(ref_vector)

        ref_hip = torch.mean(global_positions[:,:,0:1,[0,2]],dim=(1),keepdim=True)
        global_positions[...,[0,2]] = global_positions[...,[0,2]] - ref_hip
        global_positions = quaternion_apply(root_rotation, global_positions)
        global_quats = quaternion_multiply(root_rotation, global_quats)
        """ Local Space """

        return global_positions,global_quats, ref_hip,root_rotation

class TransformSeq():
    def __init__(self):
        self.trans = BatchRotateYCenterXZ()
        pass
    def transform(self,glb_pos,glb_quat):
        pos,quats, ref_hip,root_rotation = self.trans(glb_pos,glb_quat,9)
        self.ref_hip = ref_hip
        self.inv_rot = quaternion_invert(root_rotation)
        return pos,quats
    def un_transform(self,glb_pos,glb_quat):
        glb_pos = quaternion_apply(self.inv_rot,glb_pos)
        glb_rot = quaternion_multiply(self.inv_rot,glb_quat)
        glb_pos[...,[0,2]] = glb_pos[...,[0,2]]+self.ref_hip
        return glb_pos,glb_rot
class RunningLongSeq():
    def __init__(self,model,X,Q,A,S,tar_X,tar_Q,pos_offset,skeleton):
        self.window = 60
        self.source_idx = 0
        self.out_idx = 0
        self.X = X.cuda()
        self.Q = Q.cuda()

        self.pos_offset = pos_offset.cuda()
        self.tar_pos, self.tar_rot = skeleton.forward_kinematics(tar_Q[:, :120].cuda(),  self.pos_offset, tar_X[:, :120].cuda())
        self.skeleton = skeleton
        self.transform = TransformSeq()
        self.model = model.cuda()
        self.phases = model.phase_op.phaseManifold(A, S)
        self.outX = X[:,:10].cuda()
        self.outQ = Q[:,:10].cuda()
        self.outPhase = self.phases[:,:10].cuda()

        tar_id = [50,90,130,170,210,250,290,330]
        self.time = [40,40,40,40,40,80,40,40]
        get_tar_property = lambda property: [property[:,tar_id[i]-2:tar_id[i]] for i in range(len(tar_id))]
        self.X_tar = get_tar_property(self.X)
        self.Q_tar = get_tar_property(self.Q)
        self.X_tar[3][:, :, :, [0, 2]] = self.X_tar[2][:, :, :, [0]] + 20
        self.tar_id = 0
        pass
    def next_seq(self,length):
        X = self.outX[:,self.out_idx:self.out_idx+10]
        Q = self.outQ[:,self.out_idx:self.out_idx+10]
        phases = self.phases[:,self.out_idx:self.out_idx+10]#self.outPhase[:,self.out_idx:self.out_idx+10]
        X_tar = self.X_tar[self.tar_id]#torch.cat((self.X_tar[self.tar_id]),dim=1)
        Q_tar = self.Q_tar[self.tar_id]#torch.cat((self.Q_tar[self.tar_id]),dim=1)
        X = torch.cat((X,X_tar),dim=1)
        Q = torch.cat((Q,Q_tar),dim=1)
        gp,gq = self.skeleton.forward_kinematics(Q,self.pos_offset,X)
        gp,gq = self.transform.transform(gp,gq)
        gq,gp,pred_phase = synthesize(self.model,gp,gq,phases,self.tar_pos,self.tar_rot,self.pos_offset,self.skeleton,length,12)
        gp,gq = self.transform.un_transform(gp,gq)
        X,Q = self.skeleton.inverse_kinematics(gq,gp)
        self.outX = torch.cat((self.outX,X),dim=1)
        self.outQ = torch.cat((self.outQ,Q),dim=1)
        self.outPhase = torch.cat((self.outPhase,pred_phase),dim=1)
        self.out_idx += length
        self.tar_id+=1
    def iteration(self):
      #  time = [40,40,40,40,40,80,40]
        for i in range(len(self.X_tar)):
            self.next_seq(self.time[i])
    def get_source(self):
        return self.X.cpu().squeeze(0).numpy()[10:],self.Q.cpu().squeeze(0).numpy()[10:]
    def get_results(self):
        return self.outX.cpu().squeeze(0).numpy()[10:],self.outQ.cpu().squeeze(0).numpy()[10:]
    def get_target(self):
        Xs = []
        Qs = []
        for i in range(len(self.X_tar)):
            X = self.X_tar[i][0,1:]
            Q = self.Q_tar[i][0,1:]
            X = X.repeat(self.time[i],1,1)
            Q = Q.repeat(self.time[i],1,1)
            Xs.append(X)
            Qs.append(Q)
        Xs = torch.cat(Xs,dim=0)
        Qs = torch.cat(Qs,dim=0)
        return Xs.cpu().numpy(),Qs.cpu().numpy()


def synthesize(model, gp, gq, phases, tar_pos, tar_quat, pos_offset, skeleton: Skeleton, length, target_id, ifnoise=False):
    model = model.eval()
    model = model.cuda()
    #quats = Q
    offsets = pos_offset
  #  hip_pos = X
   # gp, gq = skeleton.forward_kinematics(quats, offsets, hip_pos)
    loc_rot = quat_to_or6D(gq)
    if ifnoise:
        noise = None
    else:
        noise = torch.zeros(size=(gp.shape[0], 512), dtype=gp.dtype, device=gp.device).cuda()
    tar_quat = quat_to_or6D(tar_quat)
    target_style = model.get_film_code(tar_pos.cuda(), tar_quat.cuda())  # use random style seq
    # target_style = model.get_film_code(gp.cuda(), loc_rot.cuda())
   # F = S[:, 1:] - S[:, :-1]
  #  F = model.phase_op.remove_F_discontiny(F)
   # F = F / model.phase_op.dt
   # phases = model.phase_op.phaseManifold(A, S)
    pred_pos, pred_rot, pred_phase, _ = model.shift_running(gp.cuda(), loc_rot.cuda(), phases.cuda(), None,
                                                            None,
                                                            target_style, noise, start_id=10, target_id=target_id,
                                                            length=length, phase_schedule=1.)
    pred_pos, pred_rot = pred_pos, pred_rot
    rot_pos = model.rot_to_pos(pred_rot, offsets, pred_pos[:, :, 0:1])
    pred_pos[:, :, model.rot_rep_idx] = rot_pos[:, :, model.rot_rep_idx]
    edge_len = torch.norm(offsets[:, 1:], dim=-1, keepdim=True)
    pred_pos, pred_rot = model.regu_pose(pred_pos, edge_len, pred_rot)

    GQ = skeleton.inverse_pos_to_rot(or6d_to_quat(pred_rot), pred_pos, offsets, find_secondary_axis(offsets))
    GX = skeleton.global_rot_to_global_pos(GQ, offsets, pred_pos[:, :, 0:1, :])
    phases = pred_phase[3]
    return GQ, GX, phases


if __name__ =="__main__":
    model = load_model()
    loader = load_dataSet()
    anim = BVH.read_bvh("source.bvh")
    motions = loader.train_motions['LeftHop']["BR"]
    tar_motions = loader.train_motions['Neutral']["FR"]

    def extract_property(motions):
        X = torch.from_numpy(motions['hip_pos'][0]).unsqueeze(0)
        Q = torch.from_numpy(motions['quats'][0]).unsqueeze(0)
        A = torch.from_numpy(motions['A'][0]).unsqueeze(0)/0.1
        S = torch.from_numpy(motions['S'][0]).unsqueeze(0)
        return X,Q,A,S


    X, Q, A, S = extract_property(motions)
    pos_offset = torch.from_numpy(motions['offsets'][0]).unsqueeze(0)
    with torch.no_grad():
        running_machine = RunningLongSeq(model,X,Q,A,S,X,Q,pos_offset,anim.skeleton)
        running_machine.iteration()
        anim.hip_pos,anim.quats = running_machine.get_source()
        BVH.save_bvh("source.bvh",anim)
        anim.hip_pos,anim.quats = running_machine.get_results()
        BVH.save_bvh("results.bvh",anim)
        anim.hip_pos, anim.quats = running_machine.get_target()
        BVH.save_bvh("target.bvh", anim)


