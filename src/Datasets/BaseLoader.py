import os
import pickle
import random
import numpy as np
import src.utils.BVH_mod as BVH
from src.utils.motion_process import subsample
from enum import Enum
class DataSetType(Enum):
    Train = 0
    Test = 1
'''Loader decides the data structure in the memory'''
def subsample_dict(data,ratio):
    if(ratio<=1):
        return data
    for key in data.keys():
        if(key!="offsets"):
            data[key] = data[key][::ratio,...]
    return data
class BasedLoader():
    def __init__(self,sample_ratio:int):
        self.sample_ratio = sample_ratio

    def _append(self, offsets, hip_pos, local_quat):
       # return {"offsets":[offsets],"hip_pos":[hip_pos],"quat":[local_quat]}
        return [offsets],[hip_pos],[local_quat]
        pass
    def _append_dict(self,dict,non_temporal_keys):
        return dict
    def _subsample(self,anim):
        if(self.sample_ratio>1):
            anim = subsample(anim,self.sample_ratio)
        return anim
    def load_anim(self, anim):
        anim = self._subsample(anim)
        return self._append(anim.offsets,anim.hip_pos,anim.quats)
    def load_data(self,offsets,hip_pos,quats):
        return self._append(offsets,hip_pos,quats)
    def load_dict(self,data,non_temporal_keys=["offsets"]):
        data = subsample_dict(data,self.sample_ratio)
        return self._append_dict(data,non_temporal_keys)



class WindowBasedLoader(BasedLoader):
    def __init__(self,window,overlap,subsample):
        super(WindowBasedLoader, self).__init__(subsample)
        self.window = window
        self.overlap = overlap

    def _append(self,offsets, hip_pos, local_quat):
        # Sliding windows
        step = self.window - self.overlap
        i = 0
        o=[]
        h=[]
        q=[]
        while i + self.window < local_quat.shape[0]:
            clip = lambda x: x[i:i + self.window , ...]
            o.append(offsets[np.newaxis, :, :].astype(np.float32))
            h.append(clip(hip_pos)[np.newaxis, ...].astype(np.float32))
            q.append(clip(local_quat)[np.newaxis, ...].astype(np.float32))
            i += step
        return o,h,q
    def append_dicts(self,dict,non_temporal_keys=["offsets"]):
        temporal_keys = [i for i in dict.keys() if i not in non_temporal_keys]
        length = len(dict[temporal_keys[0]])
        output = {key:[] for key in dict.keys()}
        for i in range(length):
            x = {key:dict[key][i] for key in dict.keys()}
            x = self._append_dict(x,non_temporal_keys)
            for key in dict.keys():
                output[key] = output[key]+x[key]
        return output
    def _append_dict(self,dict,non_temporal_keys=["offsets"]):
        step = self.window - self.overlap
        temporal_keys = [i for i in dict.keys() if i not in non_temporal_keys]
        length = dict[temporal_keys[0]].shape[0]
        i = 0
        output = {key:[] for key in dict.keys()}
        while i + self.window < length:
            clip = lambda x: x[i:i + self.window, ...]
            for key in temporal_keys:
                output[key].append(clip(dict[key])[np.newaxis,:,:].astype(np.float32))
            for key in non_temporal_keys:
                output[key].append(dict[key][np.newaxis,:,:].astype(np.float32))
            i += step
        return output
    def get_postfix_str(self):
        return "_"+str(self.window)+"_"+str(self.overlap)

class StreamBasedLoader(BasedLoader):
    def __init__(self,sample_ratio):
        super(StreamBasedLoader, self).__init__(sample_ratio)
        pass

'''Processor decides the training data '''
class BasedDataProcessor():
    def __init__(self):
        pass
    def _concat(self,offsets,hip_pos,quats):
        offsets = np.concatenate(offsets,axis = 0)
        local_quat = np.concatenate(quats,axis = 0)
        hip_pos = np.concatenate(hip_pos,axis = 0)
        return offsets,hip_pos,local_quat
    def __call__(self, dict,skeleton,motion_data_loader):
        offsets, hip_pos, quats = dict["offsets"], dict["hip_pos"], dict["quats"]
        return {"offsets":offsets,"hip_pos":hip_pos,"quats":quats}
'''读文件的工具类，如果从bvh文件中读取，则需要配置loader和processor，以做简单的数据处理和元数据计算'''
class MotionDataLoader():
    def __init__(self,property:dict):
       # self.dict = None

        self.loader= self.processor = None
        self.skeleton = None
        self.dataset_property = property
        self.file_prefix = []
    def setup(self,loader:BasedLoader=None,processor:BasedDataProcessor=None):
        self.loader = loader
        self.processor = processor
    def get_path(self,name:DataSetType):
        if(name==DataSetType.Train):
            return self.dataset_property["train_path"]
        elif(name == DataSetType.Test):
            return self.dataset_property['test_path']
        else:
            assert False
    def _set_dict(self,name:DataSetType,dict):
        if (name == DataSetType.Train):
            self.train_dict = dict
        else:
            self.test_dict = dict
    def get_dict(self,name:DataSetType):
        if (name == DataSetType.Train):
            return self.train_dict
        else:
            return self.test_dict
    def _process_set(self,name:DataSetType):

        path = self.get_path(name)
        remove_joints = (self.dataset_property['remove_joints'])
        if (remove_joints != None):
            remove_joints = remove_joints()
        self.files = []
        self.files = collect_bvh_files(path, self.files)
        self.file_prefix = [0]
        list_count = 0

        o = []
        h = []
        q = []
        # buffer = DataBuffer()
        for file in self.files:
            if file.endswith(".bvh"):
                offsets, hip_pos, quats = self._load_from_bvh(file,remove_joints)

                list_count += len(hip_pos)
                self.file_prefix.append(list_count)
                o+=(offsets)
                h+=(hip_pos)
                q+=(quats)



        assert list_count == len(q)
        data = {"offsets":o,"hip_pos":h,"quats":q}
        self._set_dict(name,self.processor(data,self.skeleton,self))
        return o,h,q
    def load_from_bvh_list(self,name:DataSetType):
        # data_file = os.path.join(path,"data")
        if(name == DataSetType.Test and self.get_path(DataSetType.Train)==self.get_path(DataSetType.Test)):
            assert False
        return self._process_set(name)
    def load_from_bvh_file(self,name:DataSetType,file):
        path = self.get_path(name)
        remove_joints = (self.dataset_property['remove_joints'])
        if (remove_joints != None):
            remove_joints = remove_joints()
        o = []
        h = []
        q = []
        offsets,hip_pos,quats = self._load_from_bvh(os.path.join(path,file),remove_joints)
        o += (offsets)
        h += (hip_pos)
        q += (quats)
        self._set_dict(name,self.processor(o,h,q,self.skeleton,self))
    def _load_from_bvh(self,file,remove_joints):
        if (self.loader == None or self.processor == None):
            assert False
        #print(file)
        anim = BVH.read_bvh(file, remove_joints=remove_joints, Tpose=-1, remove_gap=self.dataset_property['remove_gap'])
        if (self.skeleton == None):
            self.skeleton = anim.skeleton
        return self.loader.load_anim(anim)
    def get_skeleton(self):
        if(self.skeleton==None):
            self._load_skeleton(self.get_path(DataSetType.Train))

    def _load_skeleton(self,path):
        f = open(path +'/'+ 'skeleton', 'rb')
        self.skeleton = pickle.load(f)
        f.close()
    def _save_skeleton(self,path):
        f = open(path +'/'+ 'skeleton', 'wb+')
        pickle.dump(self.skeleton, f)
        f.close()
    def load_skeleton_only(self,name:DataSetType):
        path = self.get_path(name)
        self._load_skeleton(path)

    def save_part_to_binary(self,name:DataSetType,filename,keys):
        path = self.get_path(name)
        parseName = "train_" if name == DataSetType.Train else "test_"
        dict = self.get_dict(name)
        dict = {key:dict[key] for key in keys}
        f = open(path + '/' + parseName + filename + ".dat", "wb+")
        pickle.dump(dict, f)
        f.close()
    def load_part_from_binary(self,name:DataSetType,filename):
        path = self.get_path(name)
        parseName = "train_" if name == DataSetType.Train else "test_"
        f = open(path + '/' + parseName + filename + ".dat", 'rb')
        dict = pickle.load(f)
        f.close()
        return dict
    def save_to_binary(self,name:DataSetType,filename):
        path = self.get_path(name)
        self._save_skeleton(path)
        parseName = "train_" if name == DataSetType.Train else "test_"
        dict = self.get_dict(name)
        f = open(path+'/'+parseName+filename+".dat","wb+")
        pickle.dump(dict,f)
        f.close()

    def load_from_binary(self,name:DataSetType,filename):
        path = self.get_path(name)
        self._load_skeleton(path)
        parseName = "train_" if name == DataSetType.Train else "test_"
        f = open(path + '/' + parseName + filename + ".dat", 'rb')
        dict = pickle.load(f)
        f.close()
        self._set_dict(name,dict)







def collect_bvh_files(path,l:list):
    print(path)
    files = os.listdir(path)
    for file in files:
        file = os.path.join(path,file)
        if os.path.isdir(file):
            l = collect_bvh_files(file,l)
        elif file.endswith(".bvh"):
            l.append(file)
    return l
