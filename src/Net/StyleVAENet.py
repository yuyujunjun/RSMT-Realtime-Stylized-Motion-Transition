
import random

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

from src.Datasets.StyleVAE_DataModule import StyleVAE_DataModule
from src.Module.MoEModule import MultipleExpertsLinear
from src.Module.PhaseModule import PhaseOperator
from src.Net.CommonOperation import CommonOperator

_EPS32 = torch.finfo(torch.float32).eps
_EPS32_2 = _EPS32*2

from src.Module.VAEModule import VAE_Linear

from src.geometry.quaternions import quat_to_or6D,or6d_to_quat
from src.Datasets.BatchProcessor import BatchProcessDatav2


class MoeGateDecoder(nn.Module):
    def __init__(self,  style_dims, n_joints,n_pos_joints, condition_size, phase_dim, latent_size, num_experts):

        super(MoeGateDecoder, self).__init__()
        out_channels = 6 * n_joints + n_pos_joints*3# + phase_dim*4
        gate_in = phase_dim*2 + latent_size#+ condition_size
        self.gate = nn.Sequential(nn.Linear(gate_in, 128), nn.ELU(), nn.Linear(128, 128),nn.ELU(), nn.Linear(128, num_experts), nn.Softmax(-1))
        self.linears = nn.ModuleList([MultipleExpertsLinear(condition_size+latent_size, 512, num_experts), MultipleExpertsLinear(512+latent_size, 512, num_experts)])
        self.act = nn.ModuleList([nn.ELU(), nn.ELU(), nn.ELU()])
        self.mlp = MultipleExpertsLinear(512, out_channels, num_experts)
        self.phase_dims = phase_dim
        self.out_channels = out_channels



    def forward(self,  latent, condition, phase):
        '''input: N,C->decoder'''
        '''phase: N,C->gate network'''
        '''x: pose+latent'''

        x = condition
        coefficients = self.gate(torch.cat((phase.flatten(-2,-1),latent),dim=-1))  # self.gate(torch.cat((x,hn),dim=-1))#self.gate(torch.cat((condition,hn,phase.flatten(-2,-1)),dim=-1))#self.gate(torch.cat((hn,condition),dim=-1))##self.gate(torch.cat((phase.flatten(-2,-1),contact),dim=-1))###
        x = torch.cat((x,latent),dim=-1)
        x = self.linears[0](x,coefficients)
        x = self.act[0](x)
        x = torch.cat((x, latent), dim=-1)
        x = self.linears[1](x,coefficients)
        x = self.act[1](x)
        out = self.mlp(x,coefficients)
        pred_pose = out
        return pred_pose,coefficients
from enum import Enum
class VAEMode(Enum):
        MULTI = 1
        SINGLE = 2
class MultiVAEOperator():
    def __init__(self):
        pass
class SingleVAEOperator():
    def __init__(self):
        pass
    def shift_encode_two_frame(this,self,last_v,last_rots,nxt_v,nxt_rots,contact,last_phase,next_phase):
        co_input = torch.cat((last_v, last_rots, nxt_v, nxt_rots,last_phase.flatten(-2,-1),next_phase.flatten(-2,-1)), dim=-1)
        z,mu,log_var = self.embedding_encoder(co_input)
        return z,mu,log_var


class IAN_FilmGenerator2(nn.Module):
    def __init__(self,in_channel):
        super(IAN_FilmGenerator2, self).__init__()
        self.conv_pre = nn.Conv1d(in_channel,512,1)
        self.conv_pre2 = nn.Conv1d(512,512,3)
        self.conv0 = nn.Conv1d(512,512,3)
        self.conv1 = nn.Conv1d(512,512,5)
        self.act = nn.ReLU()
    def forward(self,x):
        # hope transformer can aggregate the sharp feature, different from the netural pose
        # from motion
        x = self.conv_pre(x)
        x = self.act(x)
        x = self.conv_pre2(x)
        x = self.act(x)
        x = self.conv0(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.act(x)
        y2 = x
        y2 = y2.transpose(1,2)
        return [y2]

class EmbeddingTwoFrameEncoder(nn.Module):
    def __init__(self,num_joints,num_pos_joints,latent_dims,phase_dims):
        super(EmbeddingTwoFrameEncoder, self).__init__()
        in_channels = (6*num_joints+6*num_pos_joints)*2 #+latent_dims
        self.linears = nn.ModuleList([nn.Linear(in_channels,512),nn.Linear(512,512),nn.Linear(512,512)])
        self.act = nn.ModuleList([nn.ELU(),nn.ELU(),nn.ELU()])
        self.mlp = VAE_Linear(512,latent_dims)
    def forward(self, condition):
        '''input: N,C->decoder'''
        '''phase: N,C->gate network'''
        '''x: pose+latent'''
        x = condition
        x = self.linears[0](x)
        x = self.act[0](x)
        x = self.linears[1](x)
        x = self.act[1](x)
        x = self.linears[2](x)
        x = self.act[2](x)
        latent,mu,log_var = self.mlp(x)
        return latent,mu,log_var

class StyleVAENet(pl.LightningModule):

    def __init__(self, skeleton, phase_dim:int=20,latent_size = 64,batch_size=64,mode="pretrain",net_mode=VAEMode.SINGLE):
        style_level_dim = [512,512]
        self.rot_rep_idx = [1, 5, 9, 10, 11, 12, 13, 15, 19]
        self.pos_rep_idx = [idx for idx in np.arange(0, skeleton.num_joints) if idx not in self.rot_rep_idx]
        '''input channel: n_joints*dimensions'''
        super(StyleVAENet, self).__init__()
        self.lr = self.init_lr = 1e-3
        self.automatic_optimization = True
        self.dt = 1./30.
        self.phase_op = PhaseOperator(self.dt)
        self.style_level_dim = style_level_dim
        self.save_hyperparameters(ignore=['mode','net_mode'])
        self.skeleton = skeleton
        self.mode = mode
        self.net_mode = net_mode

        self.vae_op = SingleVAEOperator() if net_mode==VAEMode.SINGLE else MultiVAEOperator()

        self.batch_processor = BatchProcessDatav2()
        self.embedding_encoder = EmbeddingTwoFrameEncoder(skeleton.num_joints, len(self.pos_rep_idx), latent_size,phase_dim)
        self.decoder = MoeGateDecoder(style_level_dim, skeleton.num_joints,len(self.pos_rep_idx),  9+len(self.pos_rep_idx)*6+self.skeleton.num_joints*6, phase_dim, latent_size, num_experts=8)
        self.l1Loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.common_operator = CommonOperator(batch_size=batch_size)
        self.scheduled_prob = 1.0
        self.latent_size = latent_size
        self.style_level_dim = style_level_dim
        self.sigma = 0.3
        self.initialize = True

    def transform_batch_to_VAE(self, batch):
        local_pos, local_rot = batch['local_pos'], batch['local_rot']
        edge_len = torch.norm(batch['offsets'][:, 1:], dim=-1, keepdim=True)
        return local_pos, quat_to_or6D(local_rot), edge_len, (batch['phase'])
    def kl_loss(self, mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp()) / np.prod(mu.shape)
    def shift_running(self, local_pos, local_rots, phases, As, Ss, contacts, style_code):
        '''pose: N,T,J,9'''
        '''hip: N,T,3'''
        '''phases: N,T,M'''
        '''style_code: [N,C,T,J| N,C,T,J]'''
        N, T, n_joints, C = local_pos.shape
        output_pos = torch.empty(size=(N, T - 2, n_joints, 3), device=local_pos.device)
        output_rot = torch.empty(size=(N, T - 2, n_joints, 6), device=local_rots.device)
        output_mu = torch.empty(size=(N, T - 2, self.latent_size), device=local_rots.device)
        local_pos = local_pos[:, :, self.pos_rep_idx]

        # last_S = Ss[:,0]
        output_phase = torch.empty(size=(N, T - 2, phases.shape[-2], 2), device=phases.device)
        output_A = torch.empty(size=(N, T - 2, phases.shape[-2], 1), device=phases.device)
        output_F = torch.empty(size=(N, T - 2, phases.shape[-2], 1), device=phases.device)
        output_sphase = torch.empty(size=(N, T - 2, phases.shape[-2], 2), device=phases.device)
        last_g_pos, last_g_rot = local_pos[:, 1], local_rots[:, 1]
        last_g_v = local_pos[:, 1] - local_pos[:, 0]
        last_phase = phases[:, 1]

        kl_loss = 0.
        step = 0.
        for t in range(1, T - 1):  # slice+1: we discard the first frame
            last_rel_pos = (last_g_pos - last_g_pos[:, 0:1]).flatten(-2,-1)
            last_l_v = last_g_v.flatten(-2, -1)
            last_l_rot = last_g_rot.flatten(-2,-1)
            next_rel_pos = (local_pos[:, t + 1] - local_pos[:, t+1, 0:1]).flatten(-2,-1)
            next_l_v = (local_pos[:, t + 1] - last_g_pos).flatten(-2, -1)
            next_l_rot = local_rots[:, t + 1].flatten(-2, -1)
            hip_l_v = next_l_v[..., 0:3]
            hip_l_r = next_l_rot[..., 0:6].clone()
            condition_no_style = torch.cat((last_rel_pos, last_l_v, hip_l_v, last_l_rot, hip_l_r), dim=-1)
            embedding_input = torch.cat( (last_rel_pos, next_rel_pos, last_l_v, next_l_v, last_l_rot, next_l_rot), dim=-1)
            latent, mu, log_var = self.embedding_encoder( embedding_input)
            output_mu[:, t - 1] = latent
            kl_loss = kl_loss + self.kl_loss(mu, log_var)
            step += 1
            pred_pose_, coefficients = self.decoder(latent, condition_no_style, phases[:,t+1])
            pred_l_v, pred_l_rot_v = pred_pose_[..., :len(self.pos_rep_idx) * 3].view(-1, len(self.pos_rep_idx),3), pred_pose_[..., len(self.pos_rep_idx) * 3:].view( -1, self.skeleton.num_joints, 6)
            last_l_rot = last_l_rot.view(-1, self.skeleton.num_joints, 6)
            pred_g_v = pred_l_v
            pred_g_v[:, 0] = local_pos[:, t + 1, 0] - last_g_pos[:, 0]
            pred_pos = pred_g_v + last_g_pos
            pred_rot = pred_l_rot_v + last_l_rot
            pred_rot[:, 0:1] = local_rots[:, t + 1, 0:1]

            output_pos[:, t - 1, self.pos_rep_idx] = pred_pos
            output_rot[:, t - 1] = pred_rot
            last_g_pos, last_g_rot, last_g_v = pred_pos, pred_rot, pred_g_v
        if (step > 0):
            kl_loss = kl_loss / step
        return output_pos, output_rot, kl_loss, [output_phase, output_A, output_F, output_sphase]

    def regu_pose(self, pos, edge_len, rot):
        import src.geometry.inverse_kinematics as ik
        from src.geometry.quaternions import normalized_or6d
        # In our setting, the spine idx can not be affected by this function because the bone length is constant
        pos = ik.scale_pos_to_bonelen(pos, edge_len, self.skeleton._level_joints, self.skeleton._level_joints_parents)
        rot = normalized_or6d(rot)
        return pos, rot
    def get_gt_contact(self,gt_v):
        eps = 0.5
        eps_bound = 1.
        def contact_smooth(x, idx, eps, bound):
            # x must be in range of [eps,bound]
            x = (x - eps) / (bound - eps)
            x2 = x * x
            x3 = x2 * x
            contact = 2 * (x3) - 3 * (x2) + 1
            return contact * idx
        key_joint = [3, 4, 7, 8]
        gt_fv = gt_v[:, :, key_joint]
        gt_fv = torch.sum(gt_fv.pow(2), dim=-1).sqrt()
        # first we need gt_contact
        idx = (gt_fv < eps)  # * (pred_fv>gt_fv) # get penality idx
        idx_smooth = (gt_fv >= eps) * (gt_fv < eps_bound)
        gt_contact = contact_smooth(gt_fv, idx_smooth, eps, eps_bound) + torch.ones_like(gt_fv) * idx
        return gt_contact

    def contact_foot_loss(self,contact,pred_v):
        key_joints = [3, 4, 7, 8]
        pred_fv = pred_v[:, :, key_joints]
        pred_fv = (torch.sum(pred_fv.pow(2), dim=-1) + 1e-6)  # .sqrt()
        contact_idx = torch.where(contact < 0.01, 0, contact)
        # torch.where(contact>=0.01,-contact,0) # only those who is 100% fixed, we don't apply position constrain
        contact_num = torch.count_nonzero(contact_idx)
        pred_fv = pred_fv * contact_idx
        contact_loss = pred_fv.sum() / max(contact_num, 1)
        return contact_loss
    def rot_to_pos(self,rot,offsets,hip):
        if (rot.shape[-1] == 6):
            rot = or6d_to_quat(rot)
        rot_pos = self.skeleton.global_rot_to_global_pos(rot, offsets, hip)
        return rot_pos
    def phase_loss(self,gt_phase,gt_A,gt_F,phase,A,F,sphase):
        amp_scale = 1.
        loss = {"phase": self.mse_loss(gt_phase*amp_scale,phase*amp_scale),"A":self.mse_loss(gt_A*amp_scale,A*amp_scale),"F":self.mse_loss(gt_F,F),"slerp_phase":self.mse_loss(gt_phase*amp_scale,sphase*amp_scale)}
        return loss
    def shared_forward_single(self,batch,base_epoch = 30,edge_mean =21.):
        N = batch['local_pos'].shape[0] // 2
        local_pos, local_rots, edge_len, phases = self.transform_batch_to_VAE(batch)
        A = batch['A']
        S = batch['S']

        src_code = None
        self.length = 25
        start_idx = np.random.randint(0, 60-self.length-1)
        end_idx = start_idx + self.length
        local_pos = local_pos[:, start_idx:end_idx]
        local_rots = local_rots[:, start_idx:end_idx]
        phases = phases[:, start_idx:end_idx]
        A = A[:, start_idx:end_idx]
        S = S[:, start_idx:end_idx]

        F = S[:,1:]-S[:,:-1]
        F = self.phase_op.remove_F_discontiny(F)
        F = F / self.phase_op.dt

        src_pos = local_pos[:N]
        src_rots = local_rots[:N]
        src_edge_len = edge_len[:N]
        src_phases = phases[:N]
        src_A = A[:N]
        src_F = F[:N]

        ########################################################################################
        pred_pos, pred_rot,kl, pred_phase = self.shift_running(src_pos, src_rots,src_phases, src_A, src_F,None,style_code=src_code)
        rot_pos = self.rot_to_pos(pred_rot,batch['offsets'][:N],pred_pos[:,:,0:1])
        pred_pos[:,:,self.rot_rep_idx] = rot_pos[:, :, self.rot_rep_idx]
        if (self.stage != "training" or random.random() < 0.1):
            pred_pos, pred_rot = self.regu_pose(pred_pos, src_edge_len, pred_rot)
        # in loss function, we consider the end effector's position more
        gt_contact = self.get_gt_contact(src_pos[:,3:,:] - src_pos[:,2:-1,:]).detach()
        contact_loss = self.contact_foot_loss(gt_contact,pred_pos[:,1:]-pred_pos[:,:-1])
        pos_loss = self.mse_loss(src_pos[:,2:2+self.length,:]/edge_mean,pred_pos[:,:,:]/edge_mean)
        rot_loss = self.mse_loss(src_rots[:,2:2+self.length],pred_rot)

        vae_loss = {"pos":pos_loss,"rot":rot_loss,"kl":kl,"ct":contact_loss}

        epoch = self.common_operator.get_progress(self,1,0)
        if(epoch>=1):
            vae_loss['loss'] = vae_loss['pos'] + vae_loss['rot'] + kl*0.001 + vae_loss["ct"]*0.1# + (vae_loss["phase"] + vae_loss['A'] + vae_loss['F']+ vae_loss['slerp_phase']) * 0.5
        else:
            vae_loss['loss'] = vae_loss['pos'] + vae_loss['rot'] + kl * 0.001

        return vae_loss



    def validation_step(self, batch, batch_idx):
        self.scheduled_prob = 1.
        self.scheduled_phase = 1.
        self.length = 30
        self.stage = "validation"
        loss = self.shared_forward_single(batch)
        self.common_operator.log_dict(self, loss, "val_")
        return loss['loss']

    def test_step(self, batch, batch_idx):
        self.stage = "test"
        loss = self.shared_forward_single(batch)
        self.common_operator.log_dict(self, loss, "test_")
        return loss
    def update_lr(self):
        base_epoch = 20
        if (self.mode == 'fine_tune'):
            self.scheduled_prob = 1.
            # first 20 epoch ,we increase lr to self.lr
            if (self.current_epoch < base_epoch):
                progress = self.common_operator.get_progress(self, base_epoch, 0)
                lr = self.lr * progress
            else:
                progress = self.common_operator.get_progress(self, 400, base_epoch)
                lr = (1 - progress)*self.lr+progress*1e-5
            opt = self.optimizers()
            self.common_operator.set_lr(lr, opt)
            self.log("lr", lr, logger=True)
        else:
            lr = self.lr
            #first 40 epoch, we use the original lr
            #then we decay the lr to zero until 200 epoch
            if(self.mode=='second'):
                base_epoch = 0
            progress = self.common_operator.get_progress(self, 300, base_epoch)
            decay = (1 - progress)*self.lr+progress*1e-5
            lr = decay
            opt = self.optimizers()
            self.common_operator.set_lr(lr, opt)
            self.log("lr", lr, logger=True)
    def training_step(self, batch, batch_idx):

        self.stage = "training"
        self.scheduled_prob = self.common_operator.get_progress(self,target_epoch=50,start_epoch=0)

        if(self.mode!="pretrain"):
            self.scheduled_prob = 1.

        self.update_lr()
        loss = self.shared_forward_single(batch)

        self.common_operator.log_dict(self, loss, "train_")
        self.log("prob",self.scheduled_prob,logger=True)
        self.log('prob_ph',self.scheduled_phase,logger=True)
        return loss['loss']

    def configure_optimizers(self):

        models = self.common_operator.collect_models(self)
        trained_model = []

        def weight_decay(model_name, lr, weight_decay):
            trained_model.append(model_name)
            model = getattr(self, model_name)
            return self.common_operator.add_weight_decay(model, lr, weight_decay)
        lr = self.lr

        params = weight_decay("decoder", lr, 0) + weight_decay("embedding_encoder",lr,0)
        optimizer = torch.optim.Adam(params, lr=self.lr, betas=(0.5, 0.9),amsgrad=True)

        non_train_model = [i for i in models if i not in trained_model]
        if (len(non_train_model) > 0):
            print("warning:there are some models not trained:{}".format(non_train_model))
            input("Please confirm that by Enter")
        if(self.mode=='fine_tune'):
            return [optimizer]

        else:
            return [optimizer]

from src.utils.BVH_mod import find_secondary_axis
class Application(nn.Module):
    def __init__(self,net:StyleVAENet,data_module:StyleVAE_DataModule):
        super(Application, self).__init__()
        self.Net = net
        self.data_module = data_module
        self.src_batch = None
        self.target_batch = None
        self.skeleton = data_module.skeleton

    def transform_batch(self, motion, label):
        batch = [[], label]
        batch[0] = [[motion[0], motion[1], motion[2]]]
        for i, value in enumerate(batch[0][0]):
            batch[0][0][i] = torch.from_numpy(value).unsqueeze(0).type(self.Net.dtype).to(self.Net.device)
        for key in motion[3].keys():
            motion[3][key] = torch.from_numpy(motion[3][key]).unsqueeze(0).type(self.Net.dtype).to(self.Net.device)
        batch[0][0].append(motion[3])
        batch = self.data_module.transfer_mannual(batch, 0, use_phase=True,use_sty=False)
        return batch

    def transform_anim(self, anim, label):
        batch = [[], label]
        batch[0] = [[anim.quats, anim.offsets, anim.hip_pos]]
        for i, value in enumerate(batch[0][0]):
            batch[0][0][i] = torch.from_numpy(value).unsqueeze(0).unsqueeze(0).type(self.Net.dtype).to(self.Net.device)
        batch = self.data_module.transfer_mannual(batch, 0, use_phase=False,use_sty=False)
        return batch

    def setSource(self, anim):
        if (type(anim) == tuple):
            self.src_batch = self.transform_batch(anim, 0)
        else:
            self.src_batch = self.transform_anim(anim, 0)
        self.offsets = self.src_batch['offsets']
        self.tangent = find_secondary_axis(self.offsets)
    def _get_transform_ori_motion(self, batch):
        global_pos = batch['local_pos']
        global_rot = batch['local_rot']

        # global_pos,global_rot = self.skeleton.forward_kinematics(local_rot,self.offsets,global_pos[:,:,0:1,:])
        global_rot = self.skeleton.inverse_pos_to_rot(global_rot, global_pos, self.offsets, self.tangent)
        # lp, lq = self.skeleton.inverse_kinematics((global_rot), (global_pos))
        # or
        lp, lq = self.skeleton.inverse_kinematics(global_rot, global_pos)
        return lp[0].cpu().numpy(), lq[0].cpu().numpy()

    def get_source(self):
        batch = {'local_pos':self.src_batch['local_pos'][:,16:46],"local_rot":self.src_batch['local_rot'][:,16:46]}
        return self._get_transform_ori_motion(batch)

    def draw_foot_vel(self, pos, pred_pos):
        import matplotlib.pyplot as plt
        def draw(pos,ax,key_joints):
            foot_vel = ((pos[:, 1:, key_joints] - pos[:, :-1, key_joints]) ** 2).sum(dim=-1).sqrt()
            ax.plot(foot_vel[0, :].cpu())
        fig,ax = plt.subplots(2,2,figsize=(2*1.5,2*1.))
        joints = [17,18,21,22]
        ax[0,0].set_xlabel("3")
        draw(pos,ax[0,0],joints[0])
        draw(pred_pos,ax[0,0],joints[0])

        ax[0, 1].set_xlabel("4")
        draw(pos, ax[0, 1], joints[1])
        draw(pred_pos, ax[0, 1], joints[1])

        ax[1, 0].set_xlabel("7")
        draw(pos, ax[1, 0], joints[2])
        draw(pred_pos, ax[1, 0], joints[2])

        ax[1, 1].set_xlabel("8")
        draw(pos, ax[1, 1], joints[3])
        draw(pred_pos, ax[1, 1], joints[3])

        plt.show()

    def forward(self,seed,encoding=True):
        import matplotlib.pyplot as plt
        from torch._lowrank import pca_lowrank
        self.Net.eval()
        with torch.no_grad():
            loc_pos, loc_rot, edge_len, phases= self.Net.transform_batch_to_VAE(self.src_batch)
            A = self.src_batch['A']
            S = self.src_batch['S']
            F = S[:,1:]-S[:,:-1]
            F = self.Net.phase_op.remove_F_discontiny(F)
            F = F / self.Net.phase_op.dt
            torch.random.manual_seed(seed)
            pred_pos, pred_rot,kl,pred_phase = self.Net.shift_running(loc_pos, loc_rot, phases,A,F, None,
                                                                                        None)
            def draw_projection(pred_mu):
                U, S, V = pca_lowrank(pred_mu)
                proj = torch.matmul(pred_mu, V)
                proj = proj[:, :2]
                c = np.arange(0, proj.shape[0], step=1.)
                proj = proj.cpu()
                plt.scatter(proj[:, 0], proj[:, 1], c=c)
                plt.show()

            draw_projection(phases[0].squeeze(0).flatten(-2,-1))
            loss = {}
            rot_pos = self.Net.rot_to_pos(pred_rot, self.src_batch['offsets'], pred_pos[:, :, 0:1])
            pred_pos[:, :, self.Net.rot_rep_idx] = rot_pos[:, :, self.Net.rot_rep_idx]
            # output_pos,output_rot = pred_pos,pred_rot
            output_pos, output_rot = self.Net.regu_pose(pred_pos, edge_len, pred_rot)

            loss['pos']=self.Net.mse_loss(output_pos[:]/21,loc_pos[:,2:]/21)
            loss['rot']=self.Net.mse_loss(output_rot[:,],loc_rot[:,2:])
            print(loss)
            output_pos = torch.cat((loc_pos[:, :2, ...], output_pos), dim=1)
            output_rot = torch.cat((loc_rot[:, :2, ...], output_rot), dim=1)
            output_rot = or6d_to_quat(output_rot)

            batch = {}

            batch['local_rot'] = output_rot#or6d_to_quat(output_rot)
            batch['local_pos'] = output_pos
            self.draw_foot_vel(loc_pos[:,2:], output_pos)

            return self._get_transform_ori_motion(batch)







