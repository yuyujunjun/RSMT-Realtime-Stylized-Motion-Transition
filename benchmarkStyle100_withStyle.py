import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.transforms import quaternion_multiply, quaternion_apply
from torch.utils.data import DataLoader, Dataset

import src.Datasets.BaseLoader as mBaseLoader
import src.Net.TransitionNet
from src.Datasets.BaseLoader import BasedDataProcessor
from src.Net.TransitionPhaseNet import TransitionNet_phase
from src.geometry.quaternions import quat_inv, quat_mul, quat_mul_vec, from_to_1_0_0
from src.utils.BVH_mod import Skeleton
from src.utils.np_vector import interpolate_local, remove_quat_discontinuities


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

        return global_positions,global_quats


class BenchmarkProcessor(BasedDataProcessor):
    def __init__(self):
        super(BenchmarkProcessor, self).__init__()
        # self.process = BatchRotateYCenterXZ()

    def __call__(self, dict, skeleton, motionDataLoader, ratio=1.0):
        offsets, hip_pos, quats = dict["offsets"], dict["hip_pos"], dict["quats"]
        stat = self._calculate_stat(offsets, hip_pos, quats, skeleton, 1.0)
        return {"offsets": offsets, "hip_pos": hip_pos, "quats": quats, **stat}

    def _concat(self, local_quat, offsets, hip_pos, ratio=0.2):
        local_quat = torch.from_numpy(np.concatenate((local_quat), axis=0))
        offsets = torch.from_numpy(np.concatenate((offsets), axis=0))
        hip_pos = torch.from_numpy(np.concatenate((hip_pos), axis=0))
        return local_quat, offsets, hip_pos

    def calculate_pos_statistic(self, pos):
        '''pos:N,T,J,3'''
        mean = np.mean(pos, axis=(0, 1))
        std = np.std(pos, axis=(0, 1))

        return mean, std

    def _calculate_stat(self, offsets, hip_pos, local_quat, skeleton, ratio):
        local_quat, offsets, hip_pos = self._concat(local_quat, offsets, hip_pos, ratio)
        local_quat = local_quat.to(torch.float64)
        offsets = offsets.to(torch.float64)
        hip_pos = hip_pos.to(torch.float64)
        global_positions, global_rotations = skeleton.forward_kinematics(local_quat, offsets, hip_pos)
        global_positions, global_quats, root_rotation = BatchRotateYCenterXZ().forward(global_positions,
                                                                                       global_rotations, 10)
        # global_positions = torch.cat((local_positions[:,:,0:1,:],local_positions[:,:,1:]+local_positions[:,:,0:1,:]),dim=-2)
        pos_mean, pos_std = self.calculate_pos_statistic(global_positions.cpu().numpy())
        pos_mean.astype(np.float)
        pos_std.astype(np.float)
        return {"pos_stat": [pos_mean, pos_std]}


class BenchmarkDataSet(Dataset):
    def __init__(self, data, style_keys):
        super(BenchmarkDataSet, self).__init__()
        o, h, q, a, s, b, f, style = [], [], [], [], [], [], [], []
        for style_name in style_keys:
            dict = data[style_name]['motion']
            o += dict['offsets']
            h += dict['hip_pos']
            q += dict['quats']
            a += dict['A']
            s += dict['S']
            b += dict['B']
            f += dict['F']
            for i in range(len(dict['offsets'])):
                style.append( data[style_name]['style'])


        motion = {"offsets": o, "hip_pos": h, "quats": q, "A": a, "S": s, "B": b, "F": f, "style":style}
        self.data = motion
        # motions = {"offsets": o, "hip_pos": h, "quats": q, "A": a, "S": s, "B": b, "F": f}

    def __getitem__(self, item):

        keys = ["hip_pos","quats","offsets","A","S","B","F"]
        dict = {key: self.data[key][item][0] for key in keys}
        dict["style"] = self.data["style"][item]
        return {**dict}

    def __len__(self):
        return len(self.data["offsets"])

def eval_sample(model, X, Q, x_mean, x_std, pos_offset, skeleton: Skeleton, length, param):
    # FOR EXAMPLE
    model = model.eval()
    # target frame
    target_id = None
    if "target_id" in param:
        target_id = param["target_id"]

    pred_hips = torch.zeros(size=(X.shape[0], length, 1, 3))
    pred_quats = torch.zeros(size=(X.shape[0], length, 22, 4))
    GX, GQ = skeleton.forward_kinematics(pred_quats, pos_offset, pred_hips)

    x_mean = x_mean.view(skeleton.num_joints, 3)
    x_std = x_std.view(skeleton.num_joints, 3)
    # normalized
    GX = (GX - x_mean.flatten(-2, -1)) / x_std.flatten(-2, -1)
    GX = GX.transpose(1, 2)

    return GQ, GX


def load_model():
    model_dict, function_dict, param_dict = {}, {}, {}

    model_dict['Transition'] = torch.load('./results/Transitionv2_style100/myResults/117/m_save_model_205')
    function_dict['Transition'] = src.Net.TransitionPhaseNet.eval_sample
    return model_dict, function_dict, param_dict


def catPosition(pos, pos_offset):
    pos = pos.view(pos.shape[0], pos.shape[1], 1, 3)
    return torch.cat((pos, pos_offset), 2)


def torch_fk(lrot, lpos, parents, squeeze=True):
    """
    Performs Forward Kinematics (FK) on local quaternions and local positions to retrieve global representations

    :param lrot: tensor of local quaternions with shape (..., Nb of joints, 4)
    :param lpos: tensor of local positions with shape (..., Nb of joints, 3)
    :param parents: list of parents indices
    :return: tuple of tensors of global quaternion, global positions
    """

    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]

    for i in range(1, len(parents)):
        gp.append(quat_mul_vec(gr[parents[i]], lpos[..., i:i + 1, :]) + gp[parents[i]])

        gr.append(quat_mul(gr[parents[i]], lrot[..., i:i + 1, :]))

    rot, pos = torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)
    if (squeeze):
        rot = rot.reshape(lrot.shape[0], lrot.shape[1], -1)
        pos = pos.reshape(lrot.shape[0], lrot.shape[1], -1)

    return rot, pos


def torch_quat_normalize(x, eps=1e-12):
    assert x.shape[-1] == 3 or x.shape[-1] == 4 or x.shape[-1] == 2
    # lgth =  torch.sqrt(torch.sum(x*x,dim=-1,keepdim=True))
    lgth = torch.norm(x, dim=-1, keepdim=True)
    return x / (lgth + eps)


def torch_ik_rot(grot, parents):
    return torch.cat((
        grot[..., :1, :],
        quat_mul(quat_inv(grot[..., parents[1:], :]), grot[..., 1:, :]),
    ), dim=-2)


def skating_loss(pred_seq):
    num_joints = 23
    pred_seq = pred_seq.transpose(1, 2)
    pred_seq = pred_seq.view(pred_seq.shape[0], pred_seq.shape[1], num_joints, 3)

    foot_seq = pred_seq[:, :, [3, 4, 7, 8], :]
    v = torch.sqrt(
        torch.sum((foot_seq[:, 1:, :, [0, 1, 2]] - foot_seq[:, :-1, :, [0, 1, 2]]) ** 2, dim=3, keepdim=True))
    #  v[v<1]=0
    # v=torch.abs(foot_seq[:,1:,:,[0,2]]-foot_seq[:,:-1,:,[0,2]])
    ratio = torch.abs(foot_seq[:, 1:, :, 1:2]) / 2.5  # 2.5
    exp = torch.clamp(2 - torch.pow(2, ratio), 0, 1)
    c = np.sum(exp.detach().numpy() > 0)

    s = (v * exp)
    c = max(c, 1)
    m = torch.sum(s) / c
    # m = torch.max(s)
    return m


def fast_npss(gt_seq, pred_seq):
    """
    Computes Normalized Power Spectrum Similarity (NPSS).

    This is the metric proposed by Gropalakrishnan et al (2019).
    This implementation uses numpy parallelism for improved performance.

    :param gt_seq: ground-truth array of shape : (Batchsize, Timesteps, Dimension)
    :param pred_seq: shape : (Batchsize, Timesteps, Dimension)
    :return: The average npss metric for the batch
    """
    # Fourier coefficients along the time dimension
    gt_fourier_coeffs = np.real(np.fft.fft(gt_seq, axis=1))
    pred_fourier_coeffs = np.real(np.fft.fft(pred_seq, axis=1))

    # Square of the Fourier coefficients
    gt_power = np.square(gt_fourier_coeffs)
    pred_power = np.square(pred_fourier_coeffs)

    # Sum of powers over time dimension
    gt_total_power = np.sum(gt_power, axis=1)
    pred_total_power = np.sum(pred_power, axis=1)

    # Normalize powers with totals
    gt_norm_power = gt_power / gt_total_power[:, np.newaxis, :]
    pred_norm_power = pred_power / pred_total_power[:, np.newaxis, :]

    # Cumulative sum over time
    cdf_gt_power = np.cumsum(gt_norm_power, axis=1)
    cdf_pred_power = np.cumsum(pred_norm_power, axis=1)

    # Earth mover distance
    emd = np.linalg.norm((cdf_pred_power - cdf_gt_power), ord=1, axis=1)

    # Weighted EMD
    power_weighted_emd = np.average(emd, weights=gt_total_power)

    return power_weighted_emd


def flatjoints(x):
    """
    Shorthand for a common reshape pattern. Collapses all but the two first dimensions of a tensor.
    :param x: Data tensor of at least 3 dimensions.
    :return: The flattened tensor.
    """
    return x.reshape((x.shape[0], x.shape[1], -1))


def benchmark_interpolation(models, function, params, X, Q,A,S,tar_pos,tar_quat, x_mean, x_std, offsets, skeleton: Skeleton, trans_lengths,
                            benchmarks, n_past=10, n_future=10):
    """
    Evaluate naive baselines (zero-velocity and interpolation) for transition generation on given data.
    :param X: Local positions array of shape (Batchsize, Timesteps, 1, 3)
    :param Q: Local quaternions array of shape (B, T, J, 4)
    :param x_mean : Mean vector of local positions of shape (1, J*3, 1)
    :param x_std: Standard deviation vector of local positions (1, J*3, 1)
    :param offsets: Local bone offsets tensor of shape (1, 1, J-1, 3)
    :param parents: List of bone parents indices defining the hierarchy
    :param out_path: optional path for saving the results
    :param n_past: Number of frames used as past context
    :param n_future: Number of frames used as future context (only the first frame is used as the target)
    :return: Results dictionary
    """

    # trans_lengths = [ 30,30,30,30]
    n_joints = skeleton.num_joints
    res_quat = {}
    res_pos = {}
    res_npss = {}
    res_contact = {}

    resultQ = {}
    resultX = {}

    for n_trans in trans_lengths:

        torch.cuda.empty_cache()
        print('Computing errors for transition length = {}...'.format(n_trans))
        target_id = n_trans + n_past

        # Format the data for the current transition lengths. The number of samples and the offset stays unchanged.
        curr_window = n_trans + n_past + n_future
        curr_x = X[:, :curr_window, ...]
        curr_q = Q[:, :curr_window, ...]
        # ref_quat = Q[:,curr_window-1:curr_window]

        # pos_offset = offsets.unsqueeze(1)[:,:,1:,:]
        batchsize = curr_x.shape[0]
        gt_local_quats = curr_q
        gt_roots = curr_x  # torch.unsqueeze(curr_x,dim=2)
        gt_local_poses = gt_roots  # torch.cat((gt_roots, gt_offsets), dim=2)
        global_poses, global_quats = skeleton.forward_kinematics(gt_local_quats, offsets, gt_local_poses)
        ref_quat = global_quats[:, n_past - 1:n_past]
        trans_gt_local_poses = gt_local_poses[:, n_past: n_past + n_trans, ...]
        trans_gt_local_quats = gt_local_quats[:, n_past: n_past + n_trans, ...]
        # Local to global with Forward Kinematics (FK)
        trans_gt_global_poses, trans_gt_global_quats = skeleton.forward_kinematics(trans_gt_local_quats, offsets,
                                                                                   trans_gt_local_poses)  # torch_fk(trans_gt_local_quats, trans_gt_local_poses, skeleton.parents)
        trans_gt_global_poses = trans_gt_global_poses.reshape(
            (trans_gt_global_poses.shape[0], -1, n_joints * 3)).transpose(1, 2)

        def remove_disconti(quats):
            return remove_quat_discontinuities(torch.cat((ref_quat, quats), dim=1))[:, 1:]

        trans_gt_global_quats = remove_disconti(trans_gt_global_quats)
        # Normalize

        trans_gt_global_poses = (trans_gt_global_poses - x_mean) / x_std
        # trans_gt_global_poses = trans_gt_global_poses.reshape(
        #     (trans_gt_global_poses.shape[0], -1, n_joints * 3)).transpose(1, 2)
        # Zero-velocity pos/quats
        zerov_trans_local_quats, zerov_trans_local_poses = torch.zeros_like(trans_gt_local_quats), torch.zeros_like(
            trans_gt_local_poses)
        zerov_trans_local_quats[:, :, :, :] = gt_local_quats[:, n_past - 1:n_past, :, :]
        zerov_trans_local_poses[:, :, :, :] = gt_local_poses[:, n_past - 1:n_past, :, :]
        # To global
        trans_zerov_global_poses, trans_zerov_global_quats = skeleton.forward_kinematics(zerov_trans_local_quats,
                                                                                         offsets,
                                                                                         zerov_trans_local_poses)  # torch_fk(zerov_trans_local_quats, zerov_trans_local_poses, skeleton.parents)
        trans_zerov_global_poses = trans_zerov_global_poses.reshape(
            (trans_zerov_global_poses.shape[0], -1, n_joints * 3)).transpose(1, 2)
        # Normalize
        trans_zerov_global_poses = (trans_zerov_global_poses - x_mean) / x_std
        trans_zerov_global_quats = remove_disconti(trans_zerov_global_quats)

        # trans_zerov_global_poses = trans_zerov_global_poses.reshape(
        #     (trans_zerov_global_poses.shape[0], -1, n_joints * 3)).transpose(1, 2)
        resultQ['zerov'] = trans_zerov_global_quats
        resultX['zerov'] = trans_zerov_global_poses
        # Interpolation pos/quats
        r, q = curr_x[:, :, :], curr_q
        r = r.cpu().data.numpy()
        q = q.cpu().data.numpy()

        """duration test"""
        inter_root, inter_local_quats = interpolate_local(r, q, n_trans, n_trans + n_past)
        # inter_root, inter_local_quats = interpolate_local(X.unsqueeze(dim=2).data.numpy(), Q.data.numpy(), n_trans, 30+n_past)

        inter_root = torch.Tensor(inter_root)
        inter_local_quats = torch.Tensor(inter_local_quats)

        trans_inter_root = inter_root[:, 1:-1, :, :]
        inter_local_quats = inter_local_quats[:, 1:-1, :, :]
        # To global
        trans_interp_global_poses, trans_interp_global_quats = skeleton.forward_kinematics(inter_local_quats, offsets,
                                                                                           trans_inter_root)  # torch_fk(inter_local_quats, trans_inter_local_poses, skeleton.parents)
        trans_interp_global_poses = trans_interp_global_poses.reshape(
            (trans_interp_global_poses.shape[0], -1, n_joints * 3)).transpose(1, 2)
        # Normalize
        trans_interp_global_poses = (trans_interp_global_poses - x_mean) / x_std
        # trans_interp_global_poses = trans_interp_global_poses.reshape(
        #     (trans_interp_global_poses.shape[0], -1, n_joints * 3)).transpose(1, 2)
        resultQ['interp'] = trans_interp_global_quats
        resultX['interp'] = trans_interp_global_poses
        # robust motion in between pos/quats
        # model,model_input,x_mean,x_std,pos_offset,parents,length, param)

        for key in models:
            if (key in params):
                resQ, resX = function[key](models[key], X, Q,A,S,tar_pos,tar_quat, x_mean, x_std, offsets, skeleton, n_trans, params[key])
            else:
                resQ, resX = function[key](models[key], X, Q,A,S,tar_pos,tar_quat, x_mean, x_std, offsets, skeleton, n_trans, {})
            resultQ[key] = resQ
            resultX[key] = resX

        # Local quaternion loss
        if ("gq" in benchmarks):
            for key in resultQ:
                resultQ[key] = remove_disconti(
                    resultQ[key].view(batchsize, n_trans, n_joints, 4))  # .flatten(start_dim=-2,end_dim=-1)
            trans_gt_global_quats = remove_disconti(
                trans_gt_global_quats.view(batchsize, n_trans, n_joints, 4))  # .flatten(start_dim=-2,end_dim=-1)
            for key in resultQ:
                res_quat[(key, n_trans)] = torch.mean(
                    torch.sqrt(torch.sum((resultQ[key] - trans_gt_global_quats) ** 2.0, dim=(2, 3))))
        if ("gp" in benchmarks):
            for key in resultX:
                res_pos[(key, n_trans)] = torch.mean(
                    torch.sqrt(torch.sum((resultX[key] - trans_gt_global_poses) ** 2.0, dim=1)))

        # NPSS loss on global quaternions
        if ("npss" in benchmarks):
            for key in resultQ:
                resultQ[key] = resultQ[key].cpu().data.numpy()
            for key in resultQ:
                res_npss[(key, n_trans)] = fast_npss(flatjoints(trans_gt_global_quats), flatjoints(resultQ[key]))

        # foot skating artifacts
        if ("contact" in benchmarks):
            for key in resultX:
                # resultX[key] = resultX[key].reshape(
                #     (resultX[key].shape[0], -1, n_joints , 3))
                resultX[key] = resultX[key] * x_std + x_mean
            # trans_gt_global_poses = trans_gt_global_poses.reshape(
            #     (trans_gt_global_poses.shape[0], -1, n_joints, 3))
            trans_gt_global_poses_unnomarilized = trans_gt_global_poses * x_std + x_mean
            res_contact[('gt', n_trans)] = skating_loss(trans_gt_global_poses_unnomarilized)
            for key in resultX:
                res_contact[(key, n_trans)] = skating_loss(resultX[key])

    ### === Global quat losses ===

    return res_quat, res_pos, res_npss, res_contact


def print_result(res_quat, res_pos, res_npss, res_contact, trans_lengths, out_path=None):
    def format(dt, name, value):
        s = "{0: <16}"
        for i in range(len(value)):
            s += " & {" + str(i + 1) + ":" + dt + "}"
        return s.format(name, *value)

    print()
    print("=== Global quat losses ===")
    print(format("6d", "Lengths", trans_lengths))
    for key, l in res_quat:
        if (l == trans_lengths[0]):
            avg_loss = [res_quat[(key, n)] for n in trans_lengths]
            print(format("6.3f", key, avg_loss))
    print()

    renderplot('GlobalQuatLosses', 'loss', trans_lengths, res_quat)

    ### === Global pos losses ===
    print("=== Global pos losses ===")
    print(format("6d", "Lengths", trans_lengths))
    for key, l in res_pos:
        if l == trans_lengths[0]:
            print(format("6.3f", key, [res_pos[(key, n)] for n in trans_lengths]))
    print()

    renderplot('GlobalPositionLosses', 'loss', trans_lengths, res_pos)

    ### === NPSS on global quats ===
    print("=== NPSS on global quats ===")
    print(format("6.4f", "Lengths", trans_lengths))
    for key, l in res_npss:
        if l == trans_lengths[0]:
            print(format("6.5f", key, [res_npss[(key, n)] for n in trans_lengths]))

    renderplot('NPSSonGlobalQuats', 'NPSS', trans_lengths, res_npss)

    ### === Contact loss ===
    print("=== Contact loss ===")
    print(format("6d", "Lengths", trans_lengths))
    for key, l in res_contact:
        if l == trans_lengths[0]:
            print(format("6.3f", key, [res_contact[(key, n)] for n in trans_lengths]))
    renderplot('ContactLoss', 'loss', trans_lengths, res_contact)

    # Write to file is desired
    if out_path is not None:
        res_txt_file = open(os.path.join(out_path, 'h36m_transitions_benchmark.txt'), "a")
        res_txt_file.write("=== Global quat losses ===\n")
        res_txt_file.write(format("6d", "Lengths", trans_lengths) + "\n")
        for key in res_quat:
            res_txt_file.write(format("6.3f", key, [res_quat[(key, n)] for n in trans_lengths]) + "\n")
        res_txt_file.write("\n")
        res_txt_file.write("=== Global pos losses ===" + "\n")
        res_txt_file.write(format("6d", "Lengths", trans_lengths) + "\n")
        for key in res_pos:
            res_txt_file.write(format("6.3f", key, [res_pos[(key, n)] for n in trans_lengths]) + "\n")
        res_txt_file.write("\n")
        res_txt_file.write("=== NPSS on global quats ===" + "\n")
        res_txt_file.write(format("6d", "Lengths", trans_lengths) + "\n")
        for key in res_npss:
            res_txt_file.write(format("6.3f", key, [res_npss[(key, n)] for n in trans_lengths]) + "\n")
        res_txt_file.write("\n")
        res_txt_file.write(format("6d", "Lengths", trans_lengths) + "\n")
        for key in res_contact:
            res_txt_file.write(format("6.3f", key, [res_contact[(key, n)] for n in trans_lengths]) + "\n")
        print("\n")
        res_txt_file.close()

from src.Datasets.Style100Processor import StyleLoader
def benchmarks():
    loader = mBaseLoader.WindowBasedLoader(65, 25, 1)
    # motionloader = mBaseLoader.MotionDataLoader(lafan1_property)
    style_loader = StyleLoader()
    processor = BenchmarkProcessor()
    style_loader.setup(loader, processor)
    stat_dict = style_loader.load_part_to_binary( "style100_benchmark_stat")
    mean, std = stat_dict['pos_stat']  # ,motionloader.test_dict['pos_std'] #load train x_mean, x_std  size (J * 3)
    mean, std = torch.from_numpy(mean).view(23*3,1), torch.from_numpy(std).view(23*3,1)
    style_loader.load_from_binary( "style100_benchmark_" + loader.get_postfix_str())
    #style_loader.load_from_binary( "test+phase_gv10_ori__61_21" )
    style_keys = list(style_loader.all_motions.keys())[0:90]
    dataSet = BenchmarkDataSet(style_loader.all_motions,style_keys)
    # dataLoader = DataLoader(dataSet, batch_size=len(dataSet),num_workers=0)
    dataLoader = DataLoader(dataSet, 100, num_workers=0)
    style_loader.load_skeleton_only()
    skeleton = style_loader.skeleton
    segment = 1
    num_joints = 22

    benchmarks = ["gq", "gp", "npss", "contact"]
    models, function, params = load_model()
    trans_lengths = [5, 15, 30]
    res_quat = None

    for key in models:
        for param in models[key].parameters():
            param.requires_grad = False

    iter = 0
    with torch.no_grad():
        for batch in dataLoader:

            """
            :param X: Local positions array of shape (Batchsize, Timesteps, 1, 3)
            :param Q: Local quaternions array of shape (B, T, J, 4)
            """
            iter += 1

            X = batch['hip_pos']
            A = batch['A']/0.1
            S = batch['S']
            B = batch['B']
            F = batch['F']
            gp, gq = skeleton.forward_kinematics(batch['quats'], batch['offsets'], batch['hip_pos'])
            tar_gp, tar_gq = skeleton.forward_kinematics(batch['style']['quats'], batch['style']['offsets'], batch['style']['hip_pos'])
            hip_pos = batch['hip_pos']
            local_quat = batch['quats']
            local_pos, global_quat = BatchRotateYCenterXZ().forward(gp, gq, 10)
            tar_pos, tar_quat = BatchRotateYCenterXZ().forward(tar_gp, tar_gq, 10)
            local_quat = skeleton.inverse_kinematics_quats(global_quat)
            local_quat = (remove_quat_discontinuities(local_quat.cpu()))
            hip_pos = local_pos[:, :, 0:1, :]
            quat, pos, npss, contact = benchmark_interpolation(models, function, params, hip_pos, local_quat, A, S, tar_pos, tar_quat, mean, std,
                                                               batch['offsets'], skeleton, trans_lengths, benchmarks)
            if (res_quat == None):
                res_quat, res_pos, res_npss, res_contact = quat, pos, npss, contact
            else:
                for key in res_quat:
                    res_quat[key] += quat[key]
                    res_pos[key] += pos[key]
                    res_npss[key] += npss[key]
                    res_contact[key] += contact[key]

    for key in res_quat:
        res_quat[key] /= iter
        res_pos[key] /= iter
        res_npss[key] /= iter
        res_contact[key] /= iter
    print_result(res_quat, res_pos, res_npss, res_contact, trans_lengths)


def renderplot(name, ylabel, lengths, res):
    # plt.plot(lengths, interp, label='Interp')
    for key, l in res:
        if l == lengths[0]:
            result = [res[(key, n)] for n in lengths]
            plt.plot(lengths, result, label=key)
    plt.xlabel('Lengths')
    plt.ylabel(ylabel)
    plt.title(name)
    plt.legend()
    plt.savefig(name + '.png')
    plt.close()



def duration_interpolation(models, function, params, X, Q,A,S,tar_pos,tar_quat, x_mean, x_std, offsets, skeleton: Skeleton,
                            benchmarks, trans_lengths, n_past=10, n_future=10):
    """
    Evaluate naive baselines (zero-velocity and interpolation) for transition generation on given data.
    :param X: Local positions array of shape (Batchsize, Timesteps, 1, 3)
    :param Q: Local quaternions array of shape (B, T, J, 4)
    :param x_mean : Mean vector of local positions of shape (1, J*3, 1)
    :param x_std: Standard deviation vector of local positions (1, J*3, 1)
    :param offsets: Local bone offsets tensor of shape (1, 1, J-1, 3)
    :param parents: List of bone parents indices defining the hierarchy
    :param out_path: optional path for saving the results
    :param n_past: Number of frames used as past context
    :param n_future: Number of frames used as future context (only the first frame is used as the target)
    :return: Results dictionary
    """

    # trans_lengths = [ 30,30,30,30]
    n_joints = skeleton.num_joints
    res_quat = {}
    res_pos = {}
    res_npss = {}
    res_contact = {}

    resultQ = {}
    resultX = {}
    # trans_lengths = [8,15,60,3000]
    n_trans = 30
    for length in trans_lengths:

        torch.cuda.empty_cache()
        print('Computing errors for transition length = {}...'.format(length))
        target_id = n_trans + n_past

        # Format the data for the current transition lengths. The number of samples and the offset stays unchanged.
        curr_window = n_trans + n_past + n_future
        curr_x = X[:, :curr_window, ...]
        curr_q = Q[:, :curr_window, ...]
        # ref_quat = Q[:,curr_window-1:curr_window]

        # pos_offset = offsets.unsqueeze(1)[:,:,1:,:]
        batchsize = curr_x.shape[0]
        # gt_offsets = pos_offset#.expand(curr_x.shape[0], curr_x.shape[1], num_joints - 1, 3)
        # trans_offsets=gt_offsets[:,0: 1,...].repeat(1,n_trans,1,1)
        # Ground-truth positions/quats/eulers
        gt_local_quats = curr_q
        gt_roots = curr_x  # torch.unsqueeze(curr_x,dim=2)
        gt_local_poses = gt_roots  # torch.cat((gt_roots, gt_offsets), dim=2)
        global_poses, global_quats = skeleton.forward_kinematics(gt_local_quats, offsets, gt_local_poses)
        ref_quat = global_quats[:, n_past - 1:n_past]
        trans_gt_local_poses = gt_local_poses[:, n_past: n_past + n_trans, ...]
        trans_gt_local_quats = gt_local_quats[:, n_past: n_past + n_trans, ...]
        # Local to global with Forward Kinematics (FK)
        trans_gt_global_poses, trans_gt_global_quats = skeleton.forward_kinematics(trans_gt_local_quats, offsets,
                                                                                   trans_gt_local_poses)  # torch_fk(trans_gt_local_quats, trans_gt_local_poses, skeleton.parents)
        trans_gt_global_poses = trans_gt_global_poses.reshape(
            (trans_gt_global_poses.shape[0], -1, n_joints * 3)).transpose(1, 2)

        def remove_disconti(quats):
            return remove_quat_discontinuities(torch.cat((ref_quat, quats), dim=1))[:, 1:]

        trans_gt_global_quats = remove_disconti(trans_gt_global_quats)
        # Normalize

        trans_gt_global_poses = (trans_gt_global_poses - x_mean) / x_std
        # trans_gt_global_poses = trans_gt_global_poses.reshape(
        #     (trans_gt_global_poses.shape[0], -1, n_joints * 3)).transpose(1, 2)

        # Interpolation pos/quats
        r, q = curr_x[:, :, :], curr_q
        r = r.cpu().data.numpy()
        q = q.cpu().data.numpy()

        """duration test"""
        inter_root, inter_local_quats = interpolate_local(r, q, length, n_trans + n_past)
        # inter_root, inter_local_quats = interpolate_local(X.unsqueeze(dim=2).data.numpy(), Q.data.numpy(), n_trans, 30+n_past)

        inter_root = torch.Tensor(inter_root)
        inter_local_quats = torch.Tensor(inter_local_quats)

        trans_inter_root = inter_root[:, 1:-1, :, :]
        inter_local_quats = inter_local_quats[:, 1:-1, :, :]
        # To global
        trans_interp_global_poses, trans_interp_global_quats = skeleton.forward_kinematics(inter_local_quats, offsets,
                                                                                           trans_inter_root)  # torch_fk(inter_local_quats, trans_inter_local_poses, skeleton.parents)
        trans_interp_global_poses = trans_interp_global_poses.reshape(
            (trans_interp_global_poses.shape[0], -1, n_joints * 3)).transpose(1, 2)
        # Normalize
        trans_interp_global_poses = (trans_interp_global_poses - x_mean) / x_std
        # trans_interp_global_poses = trans_interp_global_poses.reshape(
        #     (trans_interp_global_poses.shape[0], -1, n_joints * 3)).transpose(1, 2)
        resultQ['interp'] = trans_interp_global_quats
        resultX['interp'] = trans_interp_global_poses
        # robust motion in between pos/quats
        # model,model_input,x_mean,x_std,pos_offset,parents,length, param)

        for key in models:
            if (key in params):
                resQ, resX = function[key](models[key], X, Q,A,S,tar_pos,tar_quat, x_mean, x_std, offsets, skeleton, length, params[key])
            else:
                resQ, resX = function[key](models[key], X, Q,A,S,tar_pos,tar_quat, x_mean, x_std, offsets, skeleton, length, {})
            resultQ[key] = resQ
            resultX[key] = resX

        # foot skating artifacts
        if ("contact" in benchmarks):
            for key in resultX:
                # resultX[key] = resultX[key].reshape(
                #     (resultX[key].shape[0], -1, n_joints , 3))
                resultX[key] = resultX[key] * x_std + x_mean
            # trans_gt_global_poses = trans_gt_global_poses.reshape(
            #     (trans_gt_global_poses.shape[0], -1, n_joints, 3))
            trans_gt_global_poses_unnomarilized = trans_gt_global_poses * x_std + x_mean
            res_contact[('gt', length)] = skating_loss(trans_gt_global_poses_unnomarilized)
            for key in resultX:
                res_contact[(key, length)] = skating_loss(resultX[key])

    ### === Global quat losses ===

    return  res_contact

def duration_test():
    loader = mBaseLoader.WindowBasedLoader(65, 25, 1)
    # motionloader = mBaseLoader.MotionDataLoader(lafan1_property)
    style_loader = StyleLoader()
    #    processor = TransitionProcessor(10)
    processor = BenchmarkProcessor()
    style_loader.setup(loader, processor)
    stat_dict = style_loader.load_part_to_binary( "style100_benchmark_stat")
    mean, std = stat_dict['pos_stat']  # ,motionloader.test_dict['pos_std'] #load train x_mean, x_std  size (J * 3)
    mean, std = torch.from_numpy(mean).view(23*3,1), torch.from_numpy(std).view(23*3,1)
    style_loader.load_from_binary( "style100_benchmark_" + loader.get_postfix_str())
    #style_loader.load_from_binary( "test+phase_gv10_ori__61_21" )
    style_keys = list(style_loader.all_motions.keys())[0:90]
    dataSet = BenchmarkDataSet(style_loader.all_motions,style_keys)
    dataLoader = DataLoader(dataSet, batch_size=len(dataSet),num_workers=0)
    style_loader.load_skeleton_only()
    skeleton = style_loader.skeleton


    benchmarks = ["contact"]

    models, function, params = load_model()
    # trans_lengths = [8, 15, 60, 120]

    trans_lengths = [30]
    res_contact = None

    for key in models:
        for param in models[key].parameters():
            param.requires_grad = False

    iter = 0
    with torch.no_grad():
        for batch in dataLoader:

            """
            :param X: Local positions array of shape (Batchsize, Timesteps, 1, 3)
            :param Q: Local quaternions array of shape (B, T, J, 4)
            """
            iter += 1

            X = batch['hip_pos']
            A = batch['A']/0.1
            S = batch['S']
            B = batch['B']
            F = batch['F']
            gp, gq = skeleton.forward_kinematics(batch['quats'], batch['offsets'], batch['hip_pos'])
            tar_gp, tar_gq = skeleton.forward_kinematics(batch['style']['quats'], batch['style']['offsets'], batch['style']['hip_pos'])
            hip_pos = batch['hip_pos']
            local_quat = batch['quats']
            # X=X.cpu()
            # Q=Q.view(Q.shape[0],Q.shape[1],num_joints,4).cpu()
            local_pos, global_quat = BatchRotateYCenterXZ().forward(gp, gq, 10)
            local_pos[:, 12:, :, [0, 2]] = local_pos[:, 12:, :, [0, 2]] + (
                    local_pos[:, 40:41, 0:1, [0, 2]] - local_pos[:, 10:11, 0:1, [0, 2]]) * 2
            tar_pos, tar_quat = BatchRotateYCenterXZ().forward(tar_gp, tar_gq, 10)
            local_quat = skeleton.inverse_kinematics_quats(global_quat)
            local_quat = (remove_quat_discontinuities(local_quat.cpu()))
            hip_pos = local_pos[:, :, 0:1, :]
            contact = duration_interpolation(models, function, params, hip_pos, local_quat, A, S, tar_pos, tar_quat, mean, std,
                                                               batch['offsets'], skeleton, benchmarks,trans_lengths)
            if (res_contact == None):
                res_contact = contact
            else:
                for key in res_contact:
                    res_contact[key] += contact[key]

    for key in res_contact:
        res_contact[key] /= iter

    def format(dt, name, value):
        s = "{0: <16}"
        for i in range(len(value)):
            s += " & {" + str(i + 1) + ":" + dt + "}"
        return s.format(name, *value)
    print("=== Contact loss ===")
    print(format("6d", "Lengths", trans_lengths))
    for key, l in res_contact:
        if l == trans_lengths[0]:
            print(format("6.3f", key, [res_contact[(key, n)] for n in trans_lengths]))

if __name__ == '__main__':
    benchmarks()

    # duration_test()