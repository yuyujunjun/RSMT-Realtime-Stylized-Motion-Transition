import os
import src.Datasets.BaseLoader as mBaseLoader
from src.Datasets.BatchProcessor import BatchRotateYCenterXZ
import torch
import numpy as np
import random
from src.Datasets.Style100Processor import StyleLoader,Swap100StyJoints,bvh_to_binary,save_skeleton
import src.utils.BVH_mod as BVH
from src.utils.motion_process import subsample
from src.Datasets.BaseLoader import BasedDataProcessor,BasedLoader,DataSetType,MotionDataLoader,WindowBasedLoader
class TransitionProcessor(BasedDataProcessor):
    def __init__(self,ref_id):
        super(TransitionProcessor, self).__init__()
        self.process = BatchRotateYCenterXZ()
        self.ref_id = ref_id
    def __call__(self, dict, skeleton,motionDataLoader,ratio=1.0):
        offsets, hip_pos, quats = dict["offsets"], dict["hip_pos"], dict["quats"]
        stat = self._calculate_stat(offsets,hip_pos,quats,skeleton,ratio)
        return {"offsets":offsets,"hip_pos":hip_pos,"quats":quats,**stat}
    def _concat(self,local_quat,offsets,hip_pos,ratio=0.2):
        if(ratio>=1.0):
            local_quat = torch.from_numpy(np.concatenate(local_quat, axis=0))
            offsets = torch.from_numpy(np.concatenate((offsets), axis=0))
            hip_pos = torch.from_numpy(np.concatenate((hip_pos), axis=0))
        else:
            length = int(len(local_quat)*ratio)+1
            idx = []
            for i in range(length):
                idx.append(random.randrange(0,len(local_quat)))
            sample = lambda x:[x[i] for i in idx]

            local_quat = torch.from_numpy(np.concatenate(sample(local_quat), axis=0))
            offsets = torch.from_numpy(np.concatenate(sample(offsets), axis=0))
            hip_pos =torch.from_numpy(np.concatenate(sample(hip_pos), axis=0))
        return local_quat,offsets,hip_pos
    def calculate_pos_statistic(self, pos):
        '''pos:N,T,J,3'''
        mean = np.mean(pos, axis=(0, 1))
        std = np.std(pos, axis=(0, 1))
        std = np.mean(std)
        return mean, std

    def calculate_rotation_mean(self,rotation):
        mean = np.mean(rotation,axis=(0,1))
        std = np.std(rotation,axis=(0,1))
        std = np.mean(std)
        return mean ,std
    def calculate_statistic(self, local_pos,local_rot):
        pos_mean, pos_std = self.calculate_pos_statistic(local_pos[:,:,1:,:].cpu().numpy())
        vel_mean, vel_std = self.calculate_pos_statistic((local_pos[:, 1:, 1:, :] - local_pos[:, :-1, 1:, :]).cpu().numpy())
        hipv_mean, hipv_std = self.calculate_pos_statistic((local_pos[:, 1:, 0:1, :] - local_pos[:, :-1, 0:1, :]).cpu().numpy())
        rot_mean,rot_std = self.calculate_rotation_mean(local_rot[:,:,1:,:].cpu().numpy())
        rotv_mean,rotv_std = self.calculate_rotation_mean((local_rot[:,1:,1:,:]-local_rot[:,:-1,1:,:]).cpu().numpy())
        hipr_mean, hipr_std = self.calculate_rotation_mean(local_rot[:, :, 0:1, :].cpu().numpy())
        hiprv_mean, hiprv_std = self.calculate_rotation_mean((local_rot[:, 1:, 0:1, :] - local_rot[:, :-1, 0:1, :]).cpu().numpy())

        return {"pos_stat": [pos_mean, pos_std], "rot_stat":[rot_mean,rot_std],"vel_stat":[vel_mean,vel_std],
                "rotv_stat":[rotv_mean,rotv_std],"hipv_stat":[hipv_mean,hipv_std],"hipr_stat":[hipr_mean,hipr_std],"hiprv_stat":[hiprv_mean,hiprv_std]}
    def _calculate_stat(self,offsets,hip_pos,local_quat,skeleton,ratio):
        local_quat, offsets, hip_pos = self._concat(local_quat,offsets,hip_pos,ratio)
        global_positions, global_rotations = skeleton.forward_kinematics(local_quat, offsets, hip_pos)
        local_pos,local_rot = self.process(global_positions,local_quat, self.ref_id)

        return self.calculate_statistic(local_pos,local_rot)
def read_style_bvh(style,content,clip=None):
    swap_joints = Swap100StyJoints()
    anim = BVH.read_bvh(os.path.join("MotionData/100STYLE/",style,style+"_"+content+".bvh"),remove_joints=swap_joints)
    if (clip != None):
        anim.quats = anim.quats[clip[0]:clip[1], ...]
        anim.hip_pos = anim.hip_pos[clip[0]:clip[1], ...]
    anim = subsample(anim,ratio=2)
    return anim

def processStyle100Benchmark( window, overlap):
    style_loader = StyleLoader()
    processor = None
    bloader = mBaseLoader.WindowBasedLoader(window=window, overlap=overlap, subsample=1)
    style_loader.setup(bloader, processor)
    style_loader.load_dataset("+phase_gv10")

    def split_window(motions):
        for style in motions.keys():
            styles = []
            # print(style)
            if len(motions[style].keys()):
                dict = motions[style].copy()

            for content in motions[style].keys():
                motions[style][content] = bloader.append_dicts(motions[style][content])
            for content in dict.keys():
                if dict[content]['hip_pos'][0].shape[0]>=120:
                    o = dict[content]['offsets'][0]
                    h = dict[content]['hip_pos'][0][0:120]
                    q = dict[content]['quats'][0][0:120]
                    styles.append({"offsets": o, "hip_pos": h, "quats": q})
            motions[style]['style'] = styles
        result = {}
        for style_name in motions.keys():
            # print(motions.keys())
            o, h, q, a, s, b, f = [], [], [], [], [], [], []
            for content_name in motions[style_name]:
                if content_name == 'style':
                    continue
                dict = motions[style_name][content_name]
                o += dict['offsets']
                h += dict['hip_pos']
                q += dict['quats']
                a += dict['A']
                s += dict['S']
                b += dict['B']
                f += dict['F']

            # i += 1
            style = motions[style_name]['style']
            motion = {"offsets": o, "hip_pos": h, "quats": q, "A": a, "S": s, "B": b, "F": f}
            result[style_name] = {"motion":motion, "style":style}
        return result
    style_loader.test_dict = split_window(style_loader.test_motions)
    style_loader.save_to_binary("style100_benchmark_65_25", style_loader.test_dict)

def processTransitionPhaseDatasetForStyle100(window,overlap):
    style_loader = StyleLoader()
    window_loader = mBaseLoader.WindowBasedLoader(window, overlap, 1)
    processor = None #MotionPuzzleProcessor()
    style_loader.setup(window_loader, processor)
    style_loader.load_dataset("+phase_gv10")
    def split_window(motions):
        #motions = style_loader.all_motions
        for style in motions.keys():
            for content in motions[style].keys():
                motions[style][content] = window_loader.append_dicts(motions[style][content])
        return motions
    style_loader.train_motions = split_window(style_loader.train_motions)
    style_loader.test_motions = split_window(style_loader.test_motions)
    #style_loader.save_part_to_binary("motionpuzzle_statistics", ["pos_stat", "vel_stat", "rot_stat"])
    style_loader.save_dataset("+phase_gv10" + window_loader.get_postfix_str())
    print()

def processDeepPhaseForStyle100(window,overlap):
    from src.Datasets.DeepPhaseDataModule import DeepPhaseProcessor
    style_loader = StyleLoader()
    window_loader = mBaseLoader.WindowBasedLoader(window,overlap,1)
    processor = DeepPhaseProcessor(1./30)
    style_loader.setup(window_loader,processor)
    style_loader.process_from_binary()
    style_loader.save_train_test_dataset("deep_phase_gv")

def splitStyle100TrainTestSet():
    style_loader = StyleLoader()
    print("Divide the data set to train set and test set")
    style_loader.split_from_binary()
    print("argument datasets")
    style_loader.augment_dataset()
    print("down")

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--train_phase_model", action="store_true")
    parser.add_argument("--add_phase_to_dataset", action="store_true")
    parser.add_argument("--model_path",type=str,default="./results/deephase_sty/myResults/31/epoch=161-step=383778-v1.ckpt")
    parser.add_argument("--train_manifold_model", action="store_true")
    parser.add_argument("--train_sampler_model", action="store_true")
    parser.add_argument("--benchmarks", action="store_true")
    args = parser.parse_args()
    if(args.preprocess==True):
        print("######## convert all bvh files to binary files################")
        bvh_to_binary()
        save_skeleton()
        print("\nConvert down\n")
        print("Divide the dataset to train set and test set, and then argument datasets.")
        splitStyle100TrainTestSet()
    elif(args.train_phase_model==True):
        processDeepPhaseForStyle100(62,2)
    elif(args.add_phase_to_dataset==True):
        from add_phase_to_dataset import add_phase_to_100Style
        style100_info = {
            "model_path":args.model_path,
            "dt": 1. / 30,
            "window": 61
        }
        add_phase_to_100Style(style100_info)
    elif(args.train_manifold_model==True):
        processTransitionPhaseDatasetForStyle100(61,21)
    elif(args.train_sampler_model==True):
        processTransitionPhaseDatasetForStyle100(120,0)
    elif(args.benchmarks==True):
        processStyle100Benchmark(65,25)


