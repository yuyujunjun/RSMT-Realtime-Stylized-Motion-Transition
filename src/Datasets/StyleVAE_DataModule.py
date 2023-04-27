
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader

from src.Datasets.BatchProcessor import BatchRotateYCenterXZ
from src.Datasets.Style100Processor import StyleLoader
from src.Datasets.augmentation import BatchMirror
from src.Module.PhaseModule import PhaseOperator


class Style100Dataset_phase(torch.utils.data.Dataset):
    def __init__(self, dataset:dict,batch_size,is_train,keep_equal=True):
        self.is_train = is_train
        self.dataset = dataset
        self.n_styles = len(self.dataset.keys())
        self.style_ids = list(self.dataset.keys())
        self.expand_()
        self.batch_size = batch_size
        self.equal_style = keep_equal

        size = [len(self.dataset[key]) for key in self.dataset]

        if (keep_equal):
            min_size = min(size)
            min_batches = min_size//batch_size
            self.sizes = [min_batches for i in range(self.n_styles)]
            self.len = min_batches*self.n_styles
        else:

            self.sizes = [size[i]//batch_size for i in range(self.n_styles)]
            self.len = sum(self.sizes)

        self.expand_dataset = {"data":[],"sty":[]}
        self.shuffle_()
        self.style_to_idx = {}
        for i in range(len(self.style_ids)):
            self.style_to_idx[self.style_ids[i]]=i
        self.style_batch_idx = [0 for i in range(len(self.style_ids))]
        self.style_batch_start = [0]
        for i in range(len(self.sizes)):
            self.style_batch_start.append(self.sizes[i]+self.style_batch_start[i])



        pass
    def get_style_batch_for_train(self,style):
        style_id = self.style_to_idx[style]
        self.style_batch_idx[style_id] += 1
        assert(self.sizes[style_id]>0)
        if(self.style_batch_idx[style_id]>=self.sizes[style_id]):
            self.style_batch_idx[style_id] = 0
        start_idx = self.style_batch_start[style_id]
        idx = self.style_batch_idx[style_id]
        item = start_idx+idx
        return self.expand_dataset['data'][item],self.expand_dataset['sty'][item]



    def get_style_batch(self,style,style_id,batch_size):
        motions = self.dataset[style]
        length = len(motions)
        idx = np.arange(0,length)
        np.random.shuffle(idx)
        sub_idx = idx[:batch_size]
        sub_motions = [motions[j] for j in sub_idx]
        for i in range(len(sub_motions)):
            dict = {key:torch.from_numpy(sub_motions[i][3][key]).unsqueeze(0).cuda() for key in sub_motions[i][3].keys()}
            sub_motions[i] = [torch.from_numpy(sub_motions[i][j]).unsqueeze(0).cuda() for j in range(3)]+[dict]
        return {"data":sub_motions,'sty':style_id}
    def expand_(self):
        for style in self.dataset:
            motions = self.dataset[style]
            expand_ = lambda key : sum([motions[con][key] for con in motions],[])
            q = expand_('quats')
            o = expand_('offsets')
            h = expand_('hip_pos')
            A = expand_("A")
            S = expand_("S")
            B = expand_("B")
            F = expand_("F")

            self.dataset[style] = [(q[i],o[i],h[i],{"A":A[i],"S":S[i],"B":B[i],"F":F[i]}) for i in range(len(q))]#

    def shuffle_(self):
        #self.expand_dataset['data'].clear()
        #self.expand_dataset['sty'].clear()
        self.expand_dataset = {"data": [], "sty": []}
        for style_id,style in enumerate(self.dataset):
            motions = self.dataset[style]
            length = len(motions)
            idx = np.arange(0,length)
            np.random.shuffle(idx)
            sub_motions = []

            for i in range(0,len(idx),self.batch_size):
                if(i+self.batch_size>len(idx)):
                    break
                sub_idx = idx[i:i+self.batch_size]
                sub_motions.append([motions[j] for j in sub_idx])
            self.expand_dataset['data']+=(sub_motions[:self.sizes[style_id]])
            self.expand_dataset['sty']+=[self.style_ids[style_id] for i in range(self.sizes[style_id])]




    def __getitem__(self, item):


        return self.expand_dataset['data'][item],self.expand_dataset['sty'][item],self.is_train


    def __len__(self):
        return self.len

class StyleVAE_DataModule(pl.LightningDataModule):
    def __init__(self,dataloader:StyleLoader,filename,style_file_name,dt, batch_size = 32,shuffle=True,mirror=0.,use_phase = True):
        super(StyleVAE_DataModule, self).__init__()
        self.loader = dataloader
        self.data_file = filename
        self.style_file_name = style_file_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.loader.load_skeleton_only()
        self.skeleton = self.loader.skeleton
        self.processor = BatchRotateYCenterXZ()
        self.phase_op = PhaseOperator(dt)
        self.mirror = mirror

        self.batch_mirror = BatchMirror(self.skeleton, mirror_prob=mirror)
        self.use_phase = use_phase
        if(style_file_name==None):
            self.use_sty = False
        else:
            self.use_sty=True
    def prepare_data(self) -> None:
        # download, tokenize, etc
        pass

    def setup(self, stage: [str] = None) -> None:
        self.loader.load_dataset(self.data_file)
        self.train_set = Style100Dataset_phase(self.loader.train_motions, self.batch_size,True)
        self.test_set = Style100Dataset_phase(self.loader.test_motions, self.batch_size,False, keep_equal=False)
        self.val_set = self.test_set
        if(self.use_sty):
            self.loader.load_dataset(self.style_file_name)
            self.train_set_sty = Style100Dataset_phase(self.loader.train_motions, self.batch_size,True,keep_equal=False)
            self.test_set_sty = Style100Dataset_phase(self.loader.test_motions, 4,False, keep_equal=False)
            self.val_set = self.test_set

        pass

    def train_dataloader(self):
        torch.cuda.empty_cache()
        self.train_set.shuffle_()
        cons = DataLoader(self.train_set, batch_size=1, shuffle=self.shuffle, num_workers=0, drop_last=False)

        return cons

    def val_dataloader(self):  # 需要shuffle确保相同batch内的风格不相同
        cons = DataLoader(self.test_set, batch_size=1, shuffle=self.shuffle, num_workers=0, drop_last=False)

        return cons

    def test_dataloader(self):
        cons = DataLoader(self.test_set, batch_size=1, shuffle=self.shuffle, num_workers=0,
                          drop_last=False)

        return cons

    def transfer_mannual(self, batch, dataloader_idx: int, use_phase=True,use_sty=True):

        def get_data(batch, idx):
            data = [batch[0][i][idx].squeeze(1) for i in range(len(batch[0]))]
            data = torch.cat(data, dim=0)
            return data

        def get_phase(batch, str):
            data = [batch[0][i][3][str].squeeze(1) for i in range(len(batch[0]))]
            data = torch.cat(data, dim=0)
            return data

        if ('con' in batch):
            quat = torch.cat((get_data(batch['con'], 0), get_data(batch['sty'], 0)), dim=0)
            hip_pos = torch.cat((get_data(batch['con'], 2), get_data(batch['sty'], 2)), dim=0)
            offsets = torch.cat((get_data(batch['con'], 1), get_data(batch['sty'], 1)), dim=0)
            sty = [batch['con'][1], batch['sty'][1]]
            if (use_phase):
                A = torch.cat((get_phase(batch['con'], "A"), get_phase(batch['sty'], 'A')), dim=0)
                S = torch.cat((get_phase(batch['con'], "S"), get_phase(batch['sty'], 'S')), dim=0)
                A =  A/0.1
                phase = self.phase_op.phaseManifold(A, S)
            else:
                phase = None
                A = S = None


        else:
            hip_pos = get_data(batch, 2)
            quat = get_data(batch, 0)
            offsets = get_data(batch, 1)
            if (use_phase):
                A = get_phase(batch, 'A')  # ['A']
                S = get_phase(batch, 'S')  # ['S']
                A = A/0.1
                phase = self.phase_op.phaseManifold(A, S)
            else:
                A=S=None
                phase = None
            sty = [batch[1]]

        gp, gq = self.skeleton.forward_kinematics(quat, offsets, hip_pos)
        if(use_sty):
            style_name = sty[0][0]
            is_train = batch[2][0]
            if(is_train==False):# is not train
                style_batch = self.test_set_sty.get_style_batch_for_train(style_name)
            else:
                style_batch = self.train_set_sty.get_style_batch_for_train(style_name)
            assert(style_batch[1]==style_name)
            def get_style_batch(batch,idx):
                style_hip = [batch[0][i][idx] for i in range(len(batch[0]))]
                style_hip = np.concatenate(style_hip,axis=0)
                return torch.from_numpy(style_hip).to(gp.device)
            style_hip = get_style_batch(style_batch,2)
            style_quat = get_style_batch(style_batch, 0)
            style_offsets = get_style_batch(style_batch, 1)
            style_gp,style_gq = self.skeleton.forward_kinematics(style_quat,style_offsets,style_hip)
            if(is_train==False):# the test set is too small so there is not enough batch
                style_gp = style_gp.unsqueeze(0).expand((8,)+style_gp.shape).flatten(0,1)
                style_gq = style_gq.unsqueeze(0).expand((8,)+style_gq.shape).flatten(0,1)
        else:
            style_gp = style_gq = None

        local_pos, local_rot = self.processor.forward(gp, gq, 10)


        return {"local_pos": local_pos, "local_rot": local_rot, "offsets": offsets, "label": sty, "phase": phase,'A':A,'S':S,"sty_pos":style_gp,"sty_rot":style_gq}
    def on_after_batch_transfer(self, batch, dataloader_idx: int) :
        return self.transfer_mannual(batch,dataloader_idx,self.use_phase,use_sty=self.use_sty)


