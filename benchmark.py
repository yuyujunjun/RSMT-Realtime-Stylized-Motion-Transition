import time
import pickle
from src.Datasets.BatchProcessor import BatchProcessDatav2
from src.geometry.quaternions import or6d_to_quat, quat_to_or6D, from_to_1_0_0
from scipy import linalg
from pytorch3d.transforms import quaternion_multiply, quaternion_apply
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
from src.utils.BVH_mod import Skeleton, find_secondary_axis
from src.utils.np_vector import interpolate_local, remove_quat_discontinuities
from src.Datasets.Style100Processor import StyleLoader
import src.Datasets.BaseLoader as mBaseLoader
import torch
import numpy as np
import random
import torch.nn.functional as F
def load_model(args):
    model_dict, function_dict = {}, {}
    model_dict[args.model_name] = torch.load(args.model_path)
    if(hasattr(model_dict[args.model_name],"predict_phase")):
        model_dict[args.model_name].predict_phase = False
    model_dict[args.model_name].test = True
    function_dict[args.model_name] = eval_sample
    return model_dict, function_dict


def eval_sample(model, X, Q, A, S, tar_pos, tar_quat, pos_offset, skeleton: Skeleton, length, target_id, ifnoise=False):
    model = model.eval()
    model = model.cuda()
    quats = Q
    offsets = pos_offset
    hip_pos = X
    gp, gq = skeleton.forward_kinematics(quats, offsets, hip_pos)
    loc_rot = quat_to_or6D(gq)
    if ifnoise:
        noise = None
    else:
        noise = torch.zeros(size=(gp.shape[0], 512), dtype=gp.dtype, device=gp.device).cuda()
    tar_quat = quat_to_or6D(tar_quat)
    target_style = model.get_film_code(tar_pos.cuda(), tar_quat.cuda())   # use random style seq
    # target_style = model.get_film_code(gp.cuda(), loc_rot.cuda())
    F = S[:, 1:] - S[:, :-1]
    F = model.phase_op.remove_F_discontiny(F)
    F = F / model.phase_op.dt
    phases = model.phase_op.phaseManifold(A, S)

    if(hasattr(model,"predict_phase") and model.predict_phase):
        pred_pos, pred_rot, pred_phase, _,_ = model.shift_running(gp.cuda(), loc_rot.cuda(), phases.cuda(), A.cuda(),
                                                            F.cuda(),
                                                            target_style, noise, start_id=10, target_id=target_id,
                                                            length=length, phase_schedule=1.)
    else:
        pred_pos, pred_rot, pred_phase, _ = model.shift_running(gp.cuda(), loc_rot.cuda(), phases.cuda(), A.cuda(),
                                                            F.cuda(),
                                                            target_style, noise, start_id=10, target_id=target_id,
                                                            length=length, phase_schedule=1.)
    pred_pos, pred_rot = pred_pos, pred_rot
    rot_pos = model.rot_to_pos(pred_rot, offsets, pred_pos[:, :, 0:1])
    pred_pos[:, :, model.rot_rep_idx] = rot_pos[:, :, model.rot_rep_idx]
    edge_len = torch.norm(offsets[:, 1:], dim=-1, keepdim=True)
    pred_pos, pred_rot = model.regu_pose(pred_pos, edge_len, pred_rot)

    GQ = skeleton.inverse_pos_to_rot(or6d_to_quat(pred_rot), pred_pos, offsets, find_secondary_axis(offsets))
    GX = skeleton.global_rot_to_global_pos(GQ, offsets, pred_pos[:, :, 0:1, :])

    return GQ, GX



def renderplot(name, ylabel, lengths, res):
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

def calculate_fid(embeddings, gt_embeddings):
    if type(embeddings) == torch.Tensor:
        embeddings = embeddings.detach().cpu().numpy()
        gt_embeddings = gt_embeddings.detach().cpu().numpy()
    mu1 = np.mean(embeddings, axis=0)
    sigma1 = np.cov(embeddings, rowvar=False)
    mu2 = np.mean(gt_embeddings, axis=0)
    sigma2 = np.cov(gt_embeddings, rowvar=False)
    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False, blocksize=1024)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

class BatchRotateYCenterXZ(torch.nn.Module):
    def __init__(self):
        super(BatchRotateYCenterXZ, self).__init__()

    def forward(self, global_positions, global_quats, ref_frame_id):
        ref_vector = torch.cross(global_positions[:, ref_frame_id:ref_frame_id + 1, 5:6, :] - global_positions[:,
                                                                                              ref_frame_id:ref_frame_id + 1,
                                                                                              1:2, :],
                                 torch.tensor([0, 1, 0], dtype=global_positions.dtype, device=global_positions.device),
                                 dim=-1)
        root_rotation = from_to_1_0_0(ref_vector)

        # center
        ref_hip = torch.mean(global_positions[:, :, 0:1, [0, 2]], dim=(1), keepdim=True)
        global_positions[..., [0, 2]] = global_positions[..., [0, 2]] - ref_hip
        global_positions = quaternion_apply(root_rotation, global_positions)
        global_quats = quaternion_multiply(root_rotation, global_quats)

        return global_positions, global_quats

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
                style.append(random.sample(data[style_name]['style'], 1)[0])

        motion = {"offsets": o, "hip_pos": h, "quats": q, "A": a, "S": s, "B": b, "F": f, "style": style}
        self.data = motion

    def __getitem__(self, item):
        keys = ["hip_pos", "quats", "offsets", "A", "S", "B", "F"]
        dict = {key: self.data[key][item][0] for key in keys}
        dict["style"] = self.data["style"][item]
        return {**dict}

    def __len__(self):
        return len(self.data["offsets"])

def skating_loss(pred_seq):
    num_joints = 23
    # pred_seq = pred_seq.transpose(1, 2)
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

class Condition():
    def __init__(self, length, x, t):
        self.length = length
        self.x = x
        self.t = t

    def get_name(self):
        return "{}_{}_{}".format(self.length, self.x, self.t)


class GetLoss():
    def __init__(self):
        pass

    def __call__(self, data, mA, mB):
        return 0.0

def flatjoints(x):
    """
    Shorthand for a common reshape pattern. Collapses all but the two first dimensions of a tensor.
    :param x: Data tensor of at least 3 dimensions.
    :return: The flattened tensor.
    """
    return x.reshape((x.shape[0], x.shape[1], -1))

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

class GetLastRotLoss(GetLoss):
    def __init__(self):
        super(GetLastRotLoss, self).__init__()

    def __call__(self, data, mA, mB):
        a = data[mA]['rot']
        b = data[mB]['rot']
        return torch.mean(torch.sqrt(
            torch.sum((a[:, -1:, ...] - b[:, -1:, ...]) ** 2.0, dim=(2, 3))))

class GetNPSS(GetLoss):
    def __init__(self):
        super(GetNPSS, self).__init__()

    def __call__(self, data, mA, mB):
        a = data[mA]['rot']
        b = data[mB]['rot']
        return fast_npss(flatjoints(a), flatjoints(b))

class GetLastPosLoss(GetLoss):
    def __init__(self, x_mean, x_std):
        super(GetLastPosLoss, self).__init__()
        self.x_mean = x_mean
        self.x_std = x_std

    def __call__(self, data, mA, mB):
        a = data[mA]['pos'].flatten(-2, -1)
        b = data[mB]['pos'].flatten(-2, -1)
        a = (a - self.x_mean) / self.x_std
        b = (b - self.x_mean) / self.x_std
        return torch.mean(torch.sqrt(
            torch.sum((a[:, -1:, ...] - b[:, -1:, ...]) ** 2.0, dim=(2))))

class GetPosLoss(GetLoss):
    def __init__(self, x_mean, x_std):
        super(GetPosLoss, self).__init__()
        self.x_mean = x_mean
        self.x_std = x_std

    def __call__(self, data, mA, mB):
        a = data[mA]['pos'].flatten(-2, -1)
        b = data[mB]['pos'].flatten(-2, -1)
        a = (a - self.x_mean) / self.x_std
        b = (b - self.x_mean) / self.x_std
        return torch.mean(torch.sqrt(
            torch.sum((a[:, :, ...] - b[:, :, ...]) ** 2.0, dim=(2))))

class GetContactLoss(GetLoss):
    def __init__(self):
        super(GetContactLoss, self).__init__()

    def __call__(self, data, mA, mB=None):
        return skating_loss(data[mA]['pos'])


def calculate_diversity(data):
    loss_function = torch.nn.MSELoss()
    N = data.shape[-1]
    loss = 0
    for i in range(N):
        for j in range(i + 1, N):
            loss += loss_function(data[..., i], data[..., j]).detach().cpu().numpy()
    loss /= (N * N / 2)
    return loss

class GetDiversity(GetLoss):
    def __init__(self, style_seq):
        super(GetDiversity, self).__init__()
        self.style_seq = style_seq

    def __call__(self, data, mA, mB=None):
        pos = data[mA]['pos']
        DIVs = []
        start_id = 0
        for length in self.style_seq:
            if length:
                pos_seq = pos[start_id:start_id + length]
                div = calculate_diversity(pos_seq)
            else:
                div = 0
            DIVs.append(div)
            start_id += length
        dic = dict(zip(range(1, len(DIVs) + 1), DIVs))
        for key in dic.keys():
            print(dic[key])
        DIVs = np.array(DIVs)
        weight = np.array(self.style_seq)
        # mask = self.style_seq != 0
        # weight = self.style_seq[mask]
        avg_div = np.average(DIVs, weights=weight)
        return avg_div


class GetFID(GetLoss):
    def __init__(self, style_seq):
        super(GetFID, self).__init__()
        self.style_seq = style_seq
        self.norm_factor = 1
    def __call__(self, data, mA, mB):
        if mA == mB:
            return 0.0
        import time
        t = time.time()
        FIDs = []
        start_id_noaug = 0
        start_id_aug = 0
        # latentA = data[mB]['latent'][start_id:start_id + self.style_seq[0]]
        for i, length_noaug in enumerate(self.style_seq):
            if length_noaug:
                length_aug = length_noaug
                # latentA = data[mB]['latent'][start_id:start_id + length//2].flatten(1, -1)
                # latentB = data[mB]['latent'][start_id + length//2:start_id + length].flatten(1, -1)
                latentA = data[mA]['latent'][start_id_noaug:start_id_noaug + length_noaug] / self.norm_factor
                # latentA = latentA.flatten(1, -1)
                # latentA = latentA.flatten(0, 1)  # v
                latentA = latentA[:, :, :, 0].transpose(1, 2).flatten(0, 1)  # latent
                # latentA = latentA.flatten(0, 1).flatten(1, 2)  # phase
                latentB = data[mB]['latent'][start_id_aug:start_id_aug + length_aug] / self.norm_factor
                latentB = latentB[:, :, :, 0].transpose(1, 2).flatten(0, 1)  # latent
                # latentB = latentB.flatten(0, 1).flatten(1, 2)  # phase
                # latentB = latentB.flatten(0, 1)   # v
                # latentB = latentB.flatten(1, -1)
                fid = calculate_fid(latentA, latentB)
            else:
                fid = 0
            FIDs.append(fid)
            start_id_noaug += length_noaug
            start_id_aug += length_aug
        round_func = lambda x: round(x, 2)
        dic = dict(zip(range(1, len(FIDs) + 1), map(round_func, FIDs)))
        for key in dic.keys():
            print(dic[key])
        FIDs = np.array(FIDs)
        weight = np.array(self.style_seq)
        # mask = self.style_seq != 0
        # weight = self.style_seq[mask]
        avg_fid = np.average(FIDs, weights=weight)
        max_fid = np.max(FIDs)
        print(avg_fid)
        print(max_fid)
        print(f'cost : {time.time() - t:.4f}s')
        # return [avg_fid, max_fid]
        return avg_fid

class GetAcc(GetLoss):
    def __init__(self, gt_label, style_seq):
        super(GetAcc, self).__init__()
        self.style_seq = style_seq
        self.gt_label = gt_label
        pass

    def __call__(self, data, mA, mB):
        print(mA)
        start_id_noaug = 0
        correct = 0
        correct_list = []
        for i, length_noaug in enumerate(self.style_seq):

            gt_label = self.gt_label[i]
            if length_noaug:
                pred_label = data[mA]['label'][start_id_noaug:start_id_noaug + length_noaug]
                correct_num = ((pred_label == gt_label).sum()).detach().cpu().numpy()
                correct += correct_num
                correct_list.append(correct_num / length_noaug)
                start_id_noaug += length_noaug
            else:
                correct_list.append(0.)

        for i in correct_list:
            print(i)
        print('###########################')
        return correct / sum(self.style_seq)

def reconstruct_motion(models, function, X, Q, A, S, tar_pos, tar_quat, offsets, skeleton: Skeleton,
                       condition: Condition, n_past=10, n_future=10):
    # """
    # Evaluate naive baselines (zero-velocity and interpolation) for transition generation on given data.
    # :param X: Local positions array of shape (Batchsize, Timesteps, 1, 3)
    # :param Q: Local quaternions array of shape (B, T, J, 4)
    # :param x_mean : Mean vector of local positions of shape (1, J*3, 1)
    # :param x_std: Standard deviation vector of local positions (1, J*3, 1)
    # :param offsets: Local bone offsets tensor of shape (1, 1, J-1, 3)
    # :param parents: List of bone parents indices defining the hierarchy
    # :param out_path: optional path for saving the results
    # :param n_past: Number of frames used as past context
    # :param n_future: Number of frames used as future context (only the first frame is used as the target)
    # :return: Results dictionary
    # """

    data = {}

    torch.cuda.empty_cache()
    location_x = condition.x
    duration_t = condition.t
    n_trans = int(condition.length * duration_t)
    target_id = condition.length + n_past


    X[:, 12:, 0, [0, 2]] = X[:, 12:, 0, [0, 2]] + (
            X[:, target_id:target_id + 1, 0, [0, 2]] - X[:, 10:11, 0, [0, 2]]) * location_x

    # Format the data for the current transition lengths. The number of samples and the offset stays unchanged.
    curr_window = target_id + n_future
    curr_x = X[:, :curr_window, ...]
    curr_q = Q[:, :curr_window, ...]
    global_poses, global_quats = skeleton.forward_kinematics(curr_q, offsets, curr_x)



    def remove_disconti(quats):
        ref_quat = global_quats[:, n_past - 1:n_past]
        return remove_quat_discontinuities(torch.cat((ref_quat, quats), dim=1))[:, 1:]

    def get_gt_pose(n_past, target_id):
        trans_gt_global_poses, trans_gt_global_quats = global_poses[:, n_past: target_id, ...], global_quats[:,
                                                                                                n_past: target_id, ...]
        trans_gt_global_quats = remove_disconti(trans_gt_global_quats)
        return trans_gt_global_poses, trans_gt_global_quats


    gt_global_pos, gt_global_quats = get_gt_pose(n_past, target_id)
    data['gt'] = {}
    data['gt']['pos'] = gt_global_pos
    data['gt']['rot'] = gt_global_quats
    data['gt']['offsets'] = offsets

    for key in models:
        resQ, resX = function[key](models[key], X, Q, A, S, tar_pos, tar_quat, offsets, skeleton, n_trans, target_id)
        resQ = remove_disconti(resQ)
        data[key] = {}
        data[key]['rot'] = resQ
        data[key]['pos'] = resX

        # data[key]['phase'] = F.normalize(phase[3], dim=-1)
        data[key]["offsets"] = offsets

        ############################# calculate diversity #################################
        # data['gt']['phase'] = F.normalize(gt_phase, dim=-1)
        # data['div_test'] = {}
        # num_samples = 10
        # N,T,J,C = X.shape
        # X = X.repeat(num_samples,1,1,1)
        # Q = Q.repeat(num_samples,1,1,1)
        # A = A.repeat(num_samples,1,1,1)
        # S = S.repeat(num_samples,1,1,1)
        # tar_pos = tar_pos.repeat(num_samples,1,1,1)
        # tar_quat = tar_quat.repeat(num_samples,1,1,1)
        # offsets = offsets.repeat(num_samples,1,1)
        # x, q = [], []
        # resQ, resX = function[key](models[key], X, Q, A, S, tar_pos, tar_quat, offsets, skeleton, n_trans,
        #                         target_id, ifnoise=True)
        #
        # data['div_test']['pos'] = resX.view(num_samples,N,resX.shape[1],resX.shape[2],3).permute(1,2,3,4,0).cpu()
        # data['div_test']['rot'] = resQ.view(num_samples,N,resX.shape[1],resX.shape[2],4).permute(1,2,3,4,0).cpu()
        ####################################################################################

    return data, global_poses[:, n_past - 1:n_past, :, :], global_quats[:, n_past - 1:n_past, :, :]


def save_data_to_binary(path, data):
    f = open('./benchmark_result/'+ path, "wb+")
    pos = {}
    for condition in data.keys():
        pos[condition] = {}
        for method in data[condition].keys():
            pos[condition][method] = data[condition][method]['pos']
            data[condition][method]['pos'] = data[condition][method]['pos'][:,:,0:1,:]#b,t,1,3
    pickle.dump(data, f)
    for condition in data.keys():
        for method in data[condition].keys():
            data[condition][method]['pos'] = pos[condition][method]

    f.close()

def load_data_from_binary(path, skeleton:Skeleton):
    # condition{method{data}}
    f = open('./benchmark_result/'+path, "rb")
    data = pickle.load(f)
    for condition in data.keys():
        for method in data[condition].keys():
            pos = data[condition][method]['pos'] #b,t,1,3
            rot = data[condition][method]['rot'] #b,t,23,4
            offsets = data[condition][method]['offsets']
            data[condition][method]['pos'] = skeleton.global_rot_to_global_pos(rot, offsets, pos)
    f.close()
    return data

def print_result(metrics, out_path=None):
    def format(dt, name, value):
        s = "{0: <16}"
        for i in range(len(value)):
            s += " & {" + str(i + 1) + ":" + dt + "}"
        return s.format(name, *value)

    print()
    for metric_name in metrics.keys():
        if metric_name != "NPSS":
            print("=== {} losses ===".format(metric_name))
            conditions = list(metrics[metric_name].keys())
            print(format("<6", "Conditions", conditions))
            all_methods = list(metrics[metric_name][conditions[0]].keys())
            for method_name in all_methods:
                method_metric = [metrics[metric_name][condition][method_name] for condition in conditions]
                print(format("6.3f", method_name, method_metric))
        else:
            print("=== NPSS ===".format(metric_name))
            conditions = list(metrics[metric_name].keys())
            print(format("<6", "Conditions", conditions))
            all_methods = list(metrics[metric_name][conditions[0]].keys())
            for method_name in all_methods:
                method_metric = [metrics[metric_name][condition][method_name] for condition in conditions]
                # print(method_name)
                # print(method_metric)
                print(format("6.5f", method_name, method_metric))
    if out_path is not None:
        res_txt_file = open(os.path.join(out_path, 'benchmark.txt'), "a")
        for metric_name in metrics.keys():
            res_txt_file.write("\n=== {} losses ===\n".format(metric_name))
            conditions = list(metrics[metric_name].keys())
            res_txt_file.write(format("<6", "Conditions", conditions) + "\n")
            all_methods = list(metrics[metric_name][conditions[0]].keys())
            for method_name in all_methods:
                method_metric = [metrics[metric_name][condition][method_name] for condition in conditions]
                res_txt_file.write(format("6.3f", method_name, method_metric) + "\n")
        print("\n")
        res_txt_file.close()


def get_vel(pos):
    return pos[:, 1:] - pos[:, :-1]


def get_gt_latent(style_encoder, rot, pos, batch_size=1000):
    glb_vel, glb_pos, glb_rot, root_rotation = BatchProcessDatav2().forward(rot, pos)
    stard_id = 0
    data_size = glb_vel.shape[0]
    init = True
    while stard_id < data_size:
        length = min(batch_size, data_size - stard_id)
        mp_batch = {}
        mp_batch['glb_rot'] = quat_to_or6D(glb_rot[stard_id: stard_id + length]).cuda()
        mp_batch['glb_pos'] = glb_pos[stard_id: stard_id + length].cuda()
        mp_batch['glb_vel'] = glb_vel[stard_id: stard_id + length].cuda()
        latent = style_encoder.cal_latent(mp_batch).cpu()
        if init:
            output = torch.empty((data_size,) + latent.shape[1:])
            init = False
        output[stard_id: stard_id + length] = latent
        stard_id += length
    return output



def calculate_stat(conditions, dataLoader, data_size, function, models, skeleton, load_from_dict = False,data_name = None, window_size=1500):
    ##################################
    # condition{method{data}}
    if load_from_dict:
        print('load from calculated stat ...')
        t = time.time()
        outputs = load_data_from_binary(data_name, skeleton)
        print('loading data costs {} s'.format(time.time() - t))
    else:
        outputs = {}
        with torch.no_grad():
            for condition in conditions:
                print('Reconstructing motions for condition: length={}, x={}, t={} ...'.format(condition.length, condition.x,condition.t))
                start_id = 0
                outputs[condition.get_name()] = {}
                outputs[condition.get_name()] = {"gt": {}, "div_test": {}}  # , "interp":{}, "zerov":{}}
                output = outputs[condition.get_name()]
                for key in models.keys():
                    output[key] = {}
                for batch in dataLoader:

                    """
                    :param X: Local positions array of shape (Batchsize, Timesteps, 1, 3)
                    :param Q: Local quaternions array of shape (B, T, J, 4)
                    """
                    t1 = time.time()

                    A = batch['A'] / 0.1
                    S = batch['S']

                    gp, gq = skeleton.forward_kinematics(batch['quats'], batch['offsets'], batch['hip_pos'])
                    tar_gp, tar_gq = skeleton.forward_kinematics(batch['style']['quats'], batch['style']['offsets'],
                                                                 batch['style']['hip_pos'])

                    local_pos, global_quat = BatchRotateYCenterXZ().forward(gp, gq, 10)
                    tar_pos, tar_quat = BatchRotateYCenterXZ().forward(tar_gp, tar_gq, 10)
                    local_quat = skeleton.inverse_kinematics_quats(global_quat)
                    local_quat = (remove_quat_discontinuities(local_quat.cpu()))
                    hip_pos = local_pos[:, :, 0:1, :]
                    data, last_pos, last_rot = reconstruct_motion(models, function, hip_pos.cuda(), local_quat.cuda(),
                                                                  A.cuda(), S.cuda(), tar_pos.cuda(),
                                                                  tar_quat.cuda(), batch['offsets'].cuda(), skeleton,
                                                                  condition)
                    t2 = time.time()

                    if start_id == 0:
                        for method in data.keys():
                            for prop in data[method].keys():
                                output[method][prop] = torch.empty((data_size,) + data[method][prop].shape[1:])
                    for method in data.keys():
                        batch_size = data[method]['pos'].shape[0]
                        props = list(data[method].keys())
                        for prop in props:
                            output[method][prop][start_id:start_id + batch_size] = data[method][prop]
                            del data[method][prop]
                    print("batch_id : {} - {}".format(start_id, start_id + batch_size))
                    start_id += batch_size
                    print("time cost t2 - t1 : {}".format(t2 - t1))
            # print('saving data to binary file ...')
            # save_data_to_binary(data_name, outputs)
    for condition in conditions:
        if condition.length == 40:
            print('Reconstructing latent for condition: length={}, x={}, t={} ...'.format(condition.length, condition.x, condition.t))
            t = time.time()
            data = outputs[condition.get_name()]
            for method in data.keys():
                if method == "div_test":# or method == "gt":
                    continue
                start_id = 0
                length = data[method]['rot'].shape[0]
                while start_id < length:
                    window = min(window_size, length-start_id)
                    rot, pos = data[method]['rot'][start_id:start_id+window], data[method]['pos'][start_id:start_id+window]
                    # rot, pos = torch.cat((last_rot, data[method]['rot']), dim=1), torch.cat((last_pos, data[method]['pos']), dim=1)
                    # glb_vel,glb_pos,glb_rot,root_rotation = BatchProcessDatav2().forward(data[method]['rot'], data[method]['pos'])
                    glb_vel, glb_pos, glb_rot, root_rotation = BatchProcessDatav2().forward(rot, pos)
                    # data[method]["latent"] = glb_vel.flatten(-2,-1).cpu()    # use vel
                    mp_batch = {}
                    mp_batch['glb_rot'] = quat_to_or6D(glb_rot).cuda()
                    mp_batch['glb_pos'] = glb_pos.cuda()
                    mp_batch['glb_vel'] = glb_vel.cuda()
                    # latent = style_encoder.cal_latent(mp_batch).cpu() # use latent
                    # label = style_encoder.cal_label(mp_batch).cpu()
                    # if start_id == 0:
                    #     data[method]['latent'] = torch.empty((length,) + latent.shape[1:])
                    #     # data[method]['label'] = torch.empty((length,) + label.shape[1:])
                    #
                    # data[method]['latent'][start_id:start_id + window] = latent
                    # data[method]['label'][start_id:start_id + window] = label
                    # del latent
                    # del latent
                    # del label
                    start_id += window
            print('cost {} s'.format(time.time() - t))
    return outputs



def calculate_benchmark(data, benchmarks):
    metrics = {}
    for metric_key in benchmarks.keys():
        print('Computing errors for {}'.format(metric_key))
        metrics[metric_key] = {}
        if metric_key == "Diversity":
            for condition in data.keys():
                print("condition : {}".format(condition))
                metrics[metric_key][condition] = {}
                metric = benchmarks[metric_key]
                metrics[metric_key][condition]["div_test"] = metric(data[condition], "div_test")
        else:
            for condition in data.keys():
                print("condition : {}".format(condition))
                # if metric_key == 'FID' and condition.length != 40:
                #     continue
                metrics[metric_key][condition] = {}
                metric = benchmarks[metric_key]
                for method in data[condition].keys():
                    if method == "div_test":
                        continue
                    metrics[metric_key][condition][method] = metric(data[condition], method, "gt")
    return metrics

def benchmarks(args,load_stat=False, data_name = None):

    # set dataset
    style_start = 0
    style_end = 90
    batch_size = 500
    style_loader = StyleLoader()
    print('loading dataset ...')
    stat_file = 'style100_benchmark_65_25'
    style_loader.load_from_binary(stat_file)
    style_loader.load_skeleton_only()
    skeleton = style_loader.skeleton
    stat_dict = style_loader.load_part_to_binary("style100_benchmark_stat")
    mean, std = stat_dict['pos_stat']
    mean, std = torch.from_numpy(mean).view(23 * 3), torch.from_numpy(std).view(23 * 3)

    # set style
    style_keys = list(style_loader.all_motions.keys())[style_start:style_end]
    dataSet = BenchmarkDataSet(style_loader.all_motions, style_keys)

    data_size = len(dataSet)
    dataLoader = DataLoader(dataSet, batch_size=batch_size, num_workers=0, shuffle=False)
    style_seqs = [len(style_loader.all_motions[style_key]['motion']['quats']) for style_key in style_keys]

    gt_style = []

    models, function = load_model(args)
    for key in models:
        for param in models[key].parameters():
            param.requires_grad = False
    # set condition and metrics for the benchmarks
    conditions, metrics = Set_Condition_Metric(gt_style, mean, std, style_seqs)

    outputs = calculate_stat(conditions, dataLoader, data_size, function, models, skeleton, load_from_dict=load_stat, data_name=data_name)
    metrics_stat = calculate_benchmark(outputs, metrics)
    print_result(metrics_stat, './')


def Set_Condition_Metric(gt_style, mean, std, style_seqs):
    pos_loss = GetLastPosLoss(mean, std)
    gp_loss = GetPosLoss(mean,std)
    rot_loss = GetLastRotLoss()
    contact_loss = GetContactLoss()
    npss_loss = GetNPSS()
    fid = GetFID(style_seqs)
    diversity = GetDiversity(style_seqs)
    acc = GetAcc(gt_style, style_seqs)
    # set metrics
    # for example: metrics = {"Diversity" : diversity}#{"LastPos": pos_loss, "Contact": contact_loss}
    metrics = {"NPSS": npss_loss, "gp": gp_loss, "Contact": contact_loss}
    # set conditions
    conditions = [Condition(10, 0, 1), Condition(20, 0, 1), Condition(40, 0, 1),]
    return conditions, metrics


if __name__ == '__main__':
    random.seed(0)
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str,default='./results/Transitionv2_style100/myResults/117/m_save_model_205')
    parser.add_argument("--model_name",type=str,default="RSMT")
    args = parser.parse_args()
   # args.model_path = './results/Transitionv2_style100/myResults/128/m_save_model_last'
    benchmarks(args,load_stat=False, data_name='benchmark_test_data.dat')