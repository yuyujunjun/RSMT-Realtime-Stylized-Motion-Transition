import torch

import src.Datasets.BaseLoader as mBaseLoader
from src.Datasets.DeepPhaseDataModule import DeephaseDataSet, Style100DataModule
from src.Datasets.Style100Processor import StyleLoader
from src.Net.DeepPhaseNet import Application
from src.Net.DeepPhaseNet import DeepPhaseNet


class PhaseMotionStyle100Processor(mBaseLoader.BasedDataProcessor):
    def __init__(self,window,dt,model_path:str):
        from src.Datasets.DeepPhaseDataModule import DeepPhaseProcessor
        super(PhaseMotionStyle100Processor, self).__init__()
        self.processor = DeepPhaseProcessor(dt)
        #self.processor = DeepPhaseProcessorPCA(dt)
        #self.attribute = 'pos'#'gv'
        self.window = window
        self.model = DeepPhaseNet.load_from_checkpoint(model_path,style_loader=None)
    def __call__(self, dict,skeleton,motion_datalaoder= None):
        offsets, hip_pos, quats = dict["offsets"],dict["hip_pos"],dict["quats"]
        style_loader = StyleLoader()
        data_module = Style100DataModule(batch_size=32, shuffle=True, data_loader=style_loader, window_size=self.window)
        # data_module.setup()# load std
        #stat = style_loader.load_part_to_binary("deepphase_vp_statistics")
        app = Application(self.model, data_module)
        self.app = app.float()

        gv = self.processor(dict,skeleton,style_loader)['gv']
        gv = torch.from_numpy(gv).cuda()
        phase = {key:[] for key in ["A","S","B","F"]}
        h=[]
        q=[]
        o=[]
        for i in range(len(offsets)):
            print("{} in {},length:{}".format(i,len(offsets),hip_pos[i].shape[0]))
            if(hip_pos[i].shape[0]<=self.window): #gv = hip_pos[i].shape[0]-1
                continue
            dataset = DeephaseDataSet([gv[i]], self.window)
            print("dataset length: {}".format(len(dataset)))
            if(len(dataset)==0):
                continue
            self.app.Net.to("cuda")
            phases = self.app.calculate_statistic_for_dataset(dataset)
            key_frame = self.window // 2   # 61th or 31th,

            use_pos= False
            if(use_pos):
                clip = lambda x:x[key_frame:-key_frame]
            else:
                '''gv的第60帧实际上是第61帧减60，我们应该保留第61帧'''
                clip = lambda x: x[key_frame+1:-key_frame+1]
            # o[i] = clip(o[i])
            o.append(offsets[i])
            h.append(clip(hip_pos[i]))
            q.append(clip(quats[i]))
            #offsets[i]=None
            #hip_pos[i]=None
            #quats[i]=None
            for key in phases:
                phase[key].append(phases[key])

        return {"offsets": o, "hip_pos": h, "quats": q, **phase}
def add_phase_to_100Style(info):
    phase_processor = PhaseMotionStyle100Processor(info["window"], info['dt'], info["model_path"])
    bloader = mBaseLoader.StreamBasedLoader(1)
    style_loader = StyleLoader()
    style_loader.setup(bloader,mBaseLoader.BasedDataProcessor())
    style_loader.process_from_binary()
    def add_phase(motions):
        for style in motions.keys():
            print(style+"----------")
            for content in motions[style].keys():
                print(content)
                motions[style][content] = phase_processor(motions[style][content],style_loader.skeleton)
        return motions
    style_loader.train_motions = add_phase(style_loader.train_motions)
    style_loader.test_motions = add_phase(style_loader.test_motions)
    style_loader.save_dataset("+phase_gv10")

    # style_loader.process_from_binary(argument=False)
    # style_loader.train_motions = add_phase(style_loader.train_motions)
    # style_loader.test_motions = add_phase(style_loader.test_motions)
    # style_loader.save_dataset("no_augement+phase_gv10")