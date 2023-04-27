import pickle

import numpy as np
import pandas as pd
import torch

from src.Datasets.BaseLoader import BasedLoader, BasedDataProcessor
from src.utils.BVH_mod import read_bvh
from src.utils.motion_process import subsample


class Swap100StyJoints():
    def __call__(self,quats,offsets,parents,names):
        order = ["Hips",
                 "LeftHip","LeftKnee","LeftAnkle","LeftToe",
                 "RightHip","RightKnee","RightAnkle","RightToe",
                 "Chest","Chest2","Chest3","Chest4","Neck","Head",
                 "LeftCollar","LeftShoulder","LeftElbow","LeftWrist",
                 "RightCollar","RightShoulder","RightElbow","RightWrist"]
        #order = {name:i for i,name in enumerate(order)}
        ori_order = {name:i for i,name in enumerate(names)}
        tar_order = {name:i for i,name in enumerate(order)}
        n_quats = np.empty_like(quats)
        n_offsets = np.empty_like(offsets)
        n_parents = parents.copy()
        for i,name in enumerate(order):
            or_idx = ori_order[name]
            n_quats[:, i] = quats[:, or_idx]
            n_offsets[ i] = offsets[ or_idx]
            if(i!=or_idx):
                par_name = names[parents[or_idx]]
                par_id = tar_order[par_name]
            else:
                par_id = parents[or_idx]
            n_parents[i] = par_id
        return n_quats,n_offsets,n_parents,order



#dataset_list = pd.read_csv(root_dir+"Dataset_List.csv")
#anim = read_bvh("./MotionData/lafan1/train/aiming1_subject4.bvh")
#root_dir = "./MotionData/100STYLE/"
#anim = read_bvh(root_dir+"Aeroplane/Aeroplane_BR.bvh",remove_joints=Swap100StyJoints())
#save_bvh("example.bvh",anim)
def bvh_to_binary():
    root_dir = "./MotionData/100STYLE/"
    frame_cuts = pd.read_csv(root_dir + "Frame_Cuts.csv")
    n_styles = len(frame_cuts.STYLE_NAME)
    style_name = [frame_cuts.STYLE_NAME[i] for i in range(n_styles)]
    content_name = ["BR", "BW", "FR", "FW", "ID", "SR", "SW", "TR1", "TR2", "TR3"]

    def extractSeqRange(start,end):
        start = start.astype('Int64')
        end = end.astype('Int64')

        return [[(start[i]),(end[i])] for i in range(len(start))]
    content_range = {name:extractSeqRange(frame_cuts[name+"_START"],frame_cuts[name+"_STOP"]) for name in content_name}
    def clip_anim(anim,start,end):
        anim.quats = anim.quats[start:end]
        anim.hip_pos = anim.hip_pos[start:end]
        return anim
    for i in range(n_styles):
        anim_style = {}
        folder = root_dir + style_name[i] + "/"
        for content in content_name:
            ran = content_range[content][i]
            if(type(content_range[content][i][0])!=type(pd.NA)):
                file = folder+style_name[i]+"_"+content+".bvh"
                anim = read_bvh(file,remove_joints=Swap100StyJoints())
                anim = clip_anim(anim,ran[0],ran[1])
                anim = subsample(anim,2)
                anim_style[content] = {"quats":anim.quats.astype(np.float32),"offsets":anim.offsets.astype(np.float32),"hips":anim.hip_pos.astype(np.float32)}
        f = open(folder+"binary.dat","wb+")
        pickle.dump(anim_style,f)
        f.close()
def save_skeleton():
    root_dir = "./MotionData/100STYLE/"
    anim = read_bvh(root_dir+"Aeroplane/Aeroplane_BR.bvh",remove_joints=Swap100StyJoints())
    f = open(root_dir+"skeleton","wb+")
    pickle.dump(anim.skeleton,f)
    f.close()
def read_binary():
    root_dir = "./MotionData/100STYLE/"
    frame_cuts = pd.read_csv(root_dir + "Frame_Cuts.csv")
    n_styles = len(frame_cuts.STYLE_NAME)
    style_name = [frame_cuts.STYLE_NAME[i] for i in range(n_styles)]
    content_name = ["BR", "BW", "FR", "FW", "ID", "SR", "SW", "TR1", "TR2", "TR3"]
    motion_styles={}
    for i in range(n_styles):
        #anim_style = {}
        folder = root_dir + style_name[i] + "/"
        f = open(folder + "binary.dat", "rb")
        anim_style = pickle.load(f)
        f.close()
        motion_styles[style_name[i]]=anim_style
    return motion_styles,style_name
#bvh_to_binary()
#save_skeleton()


class StyleLoader():
    def __init__(self,root_dir = "./MotionData/100STYLE/"):
        self.root_dir = root_dir

    def setup(self,loader:BasedLoader,processor:BasedDataProcessor):
        self.loader = loader
        self.processor = processor
    def save_part_to_binary(self,filename,keys):
        path = self.root_dir
        dict = {key:self.train_dict[key] for key in keys}
        f = open(path + '/' + filename + ".dat", "wb+")
        pickle.dump(dict, f)
        f.close()
    def load_part_to_binary(self,filename):
        path = "./"#self.root_dir
        f = open(path + '/' + filename + ".dat", "rb")
        stat = pickle.load(f)
        f.close()
        return stat
    # dataset: all motions, the motions are splited into windows
    def save_dataset(self,filename):
        path = self.root_dir
        f = open(path + '/train' + filename + ".dat", "wb")
        pickle.dump(self.train_motions, f)
        f.close()
        f = open(path + '/test' + filename + ".dat", "wb")
        pickle.dump(self.test_motions, f)
        f.close()
    def load_dataset(self,filename):
        path = self.root_dir
        f = open(path + '/train' + filename + ".dat", "rb")
        self.train_motions = pickle.load(f)
        f.close()
        f = open(path + '/test' + filename + ".dat", "rb")
        self.test_motions = pickle.load(f)
        f.close()
    # save train set and test set
    def save_train_test_dataset(self,filename):
        path = self.root_dir
        dict = {"train": self.train_dict, "test": self.test_dict}
        f = open(path+'/'+filename+".dat","wb+")
        pickle.dump(dict,f)
        f.close()
    def load_train_test_dataset(self,filename):
        path = self.root_dir
        f = open(path + '/' + filename + ".dat", "rb")
        dict = pickle.load( f)#{"train": self.train_dict, "test": self.test_dict}
        f.close()
        self.train_dict = dict['train']
        self.test_dict = dict['test']

    def load_style_code(self,filename):
        path = self.root_dir
        f = open(path + '/' + filename + "_stycode.dat", "rb")
        self.style_codes = pickle.load(f)
        f.close()
    def save_style_code(self,filename,style_codes):
        path = self.root_dir
        f = open(path + '/' + filename + "_stycode.dat", "wb+")
        pickle.dump(style_codes, f)
        f.close()
    def load_from_binary(self,filename):
        path = self.root_dir
        f = open(path + '/' + filename + ".dat", "rb")
        self.all_motions = pickle.load(f)
        f.close()
    def save_to_binary(self,filename,all_motions):
        path = self.root_dir
        f = open(path + '/' + filename + ".dat", "wb+")
        pickle.dump(all_motions, f)
        f.close()
    def augment_dataset(self):
        from src.Datasets.augmentation import TemporalScale,BatchMirror
        folder = "./MotionData/100STYLE/"
        self._load_skeleton(folder)
        mirror = BatchMirror(self.skeleton,1.)
        scale = TemporalScale(1.)

        def augment_motions(motions):
            for style in motions.keys():
                content_keys = list(motions[style].keys())
                for content in content_keys:
                    seq = motions[style][content]
                    quats = torch.from_numpy(seq['quats']).unsqueeze(0).float().cuda()
                    offsets = torch.from_numpy(seq['offsets']).unsqueeze(0).float().cuda()
                    hips = torch.from_numpy(seq['hips']).unsqueeze(0).float().cuda()
                    # mirror
                    gp,gq = self.skeleton.forward_kinematics(quats,offsets,hips)
                    gp,gq = mirror(gp,gq)
                    mirror_hips,mirror_quats = self.skeleton.inverse_kinematics(gq,gp)
                    mirror_hips,mirror_quats = mirror_hips.squeeze(0).cpu().numpy(),mirror_quats.squeeze(0).cpu().numpy()
                    motions[style]["mr_"+content] = {"quats":mirror_quats,"offsets":seq['offsets'],"hips":mirror_hips}
                    # scale
                    sc_hips,sc_quats = scale(hips,quats)
                    sc_hips, sc_quats = sc_hips.squeeze(0).cpu().numpy(), sc_quats.squeeze(0).cpu().numpy()
                    motions[style]["sca_" + content] = {"quats": sc_quats, "offsets": seq['offsets'], "hips": sc_hips}
            return motions

        f = open(folder + "train_binary.dat", 'rb')
        self.train_motions = pickle.load(f)
        f.close()
        f = open(folder + 'test_binary.dat', 'rb')
        self.test_motions = pickle.load(f)
        f.close()

        self.train_motions = augment_motions(self.train_motions)
        self.test_motions = augment_motions(self.test_motions)
        f = open(folder + "train_binary_agument.dat", "wb")
        pickle.dump(self.train_motions, f)
        f.close()
        f = open(folder + "test_binary_agument.dat", "wb")
        pickle.dump(self.test_motions, f)
        f.close()



    def split_from_binary(self):
        folder = "./MotionData/100STYLE/"
        self.load_skeleton_only()
        self.all_motions, self.style_names = read_binary()

        self.train_motions = {}
        self.test_motions = {}
        # 镜像数据集：
        from src.Datasets.BatchProcessor import BatchMirror
        batch_mirror = BatchMirror(self.skeleton, 1.)
        for style in self.style_names[:-10]:
            self.train_motions[style]={}
            self.test_motions[style]={}
            for content in self.all_motions[style].keys():
                seq = self.all_motions[style][content]

                length = seq['quats'].shape[0]
                if(length>2000):
                    test_length = length//10
                    self.train_motions[style][content]={}
                    self.train_motions[style][content]['quats'] = seq['quats'][:-test_length]
                    self.train_motions[style][content]['offsets'] = seq['offsets']#[:-test_length]
                    self.train_motions[style][content]['hips'] = seq['hips'][:-test_length]
                    self.test_motions[style][content]={}
                    self.test_motions[style][content]['quats'] = seq['quats'][-test_length:]
                    self.test_motions[style][content]['offsets'] = seq['offsets']#[-test_length:]
                    self.test_motions[style][content]['hips'] = seq['hips'][-test_length:]
                else:
                    self.train_motions[style][content] = seq
        for style in self.style_names[-10:]:
            self.test_motions[style] = self.all_motions[style]



        f = open(folder + "train_binary.dat", "wb")
        pickle.dump(self.train_motions,f)
        f.close()
        f = open(folder+"test_binary.dat","wb")
        pickle.dump(self.test_motions,f)
        f.close()
    # read from each file
    def process_from_binary(self,argument = True):
        folder = "./MotionData/100STYLE/"
        if(argument):
            f = open(folder+"train_binary_agument.dat",'rb')
        else:
            f = open(folder+"train_binary.dat",'rb')
        self.train_motions = pickle.load(f)
        f.close()
        if (argument):
            f = open(folder + "test_binary_agument.dat", 'rb')
        else:
            f = open(folder + "test_binary.dat", 'rb')

        self.test_motions = pickle.load(f)
        f.close()
        self.train_dict = self._process_dataset(self.train_motions)
        self.test_dict = self._process_dataset(self.test_motions)
        print("process done")
        #self.loader.load_data()

    def _process_dataset(self,motions):
        o, h, q, s = [], [], [], []
        for style_name in motions.keys():
            for content_name in motions[style_name]:
                dict = motions[style_name][content_name]
                off, hip, quat = self.loader.load_data(dict['offsets'], dict['hips'], dict['quats'])
                dict['offsets'], dict['hip_pos'], dict['quats'] = off, hip, quat
                del dict['hips']
                o += off
                h += hip
                q += quat
                s += [style_name for i in range(2*len(off))]
                # motions[style_name][content_name] = dict
        self.load_skeleton_only()

        train_set = self.processor({"offsets": o, "hip_pos": h, "quats": q}, self.skeleton, self)

        train_set["style"] = s
        return train_set
    def _load_skeleton(self,path):
        f = open(path +'/'+ 'skeleton', 'rb')
        self.skeleton = pickle.load(f)
        f.close()
    def load_skeleton_only(self):
        self._load_skeleton(self.root_dir)


