
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.utilities.seed import seed_everything

from src.Datasets.DeepPhaseDataModule import Style100DataModule
from src.Datasets.Style100Processor import StyleLoader, Swap100StyJoints
from src.Net.DeepPhaseNet import DeepPhaseNet, Application
from src.utils import BVH_mod as BVH
from src.utils.locate_model import locate_model
from src.utils.motion_process import subsample


#from src.Datasets.DataSetProperty import lafan1_property,cmu_property
def setup_seed(seed:int):
    seed_everything(seed,True)
def test_model():
    dict = {}
    dict['limit_train_batches'] = 1.
    dict['limit_val_batches'] = 1.
    return dict
def detect_nan_par():
    '''track_grad_norm": 'inf'''
    return { "detect_anomaly":True}
def select_gpu_par():
    return {"accelerator":'gpu', "auto_select_gpus":True, "devices":-1}

def create_common_states(prefix:str):
    log_name = prefix+'/'
    '''test upload'''
    parser = ArgumentParser()
    parser.add_argument("--dev_run", action="store_true")
    parser.add_argument("--version", type=str, default="-1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n_phases",type=int,default=10)
    parser.add_argument("--epoch",type=str,default = '')
    parser.add_argument("--test",action="store_true")
    args = parser.parse_args()
    ckpt_path = "results/"
    if (args.version != "-1"):
        version = args.version
    else:
        version = None
    '''Create Loggers tensorboard'''
    if args.dev_run:
        log_name += "dev_run"
    else:
        log_name += "myResults"
    tb_logger = pl.loggers.TensorBoardLogger(save_dir="tensorboard_logs/", name=log_name, version=version)

    ckpt_path = os.path.join(ckpt_path, log_name, str(tb_logger.version))
    if (args.resume == True):
        resume_from_checkpoint = os.path.join(os.path.join(ckpt_path, "last.ckpt"))  # results/version/last.ckpt"
    else:
        resume_from_checkpoint = None
    checkpoint_callback = [ModelCheckpoint(dirpath=ckpt_path + "/", save_top_k=-1, save_last=True, every_n_epochs=1,save_weights_only=True),
                           ModelCheckpoint(dirpath=ckpt_path + "/", save_top_k=1, monitor="val_loss", save_last=False, every_n_epochs=1)]
    '''Train'''
    checkpoint_callback[0].CHECKPOINT_NAME_LAST = "last"
    profiler = SimpleProfiler()
    trainer_dict = {
        "callbacks":checkpoint_callback,
        "profiler":profiler,
        "logger":tb_logger
    }
    return args,trainer_dict,resume_from_checkpoint,ckpt_path

def read_style_bvh(style,content,clip=None):
    swap_joints = Swap100StyJoints()
    anim = BVH.read_bvh(os.path.join("MotionData/100STYLE/",style,style+"_"+content+".bvh"),remove_joints=swap_joints)
    if (clip != None):
        anim.quats = anim.quats[clip[0]:clip[1], ...]
        anim.hip_pos = anim.hip_pos[clip[0]:clip[1], ...]
    anim = subsample(anim,ratio=2)
    return anim

def training_style100():
    args, trainer_dict, resume_from_checkpoint, ckpt_path = create_common_states("deephase_sty")
    '''Create the model'''
    frequency = 30
    window = 61

    style_loader = StyleLoader()
    batch_size = 32
    data_module = Style100DataModule( batch_size=batch_size,shuffle=True,data_loader=style_loader,window_size=window)
    model = DeepPhaseNet(args.n_phases, data_module.skeleton, window, 1.0 / frequency,batch_size=batch_size)  # or model = pl.LightningModule().load_from_checkpoint(PATH)
    if (args.test == False):
        if (args.dev_run):
            trainer = Trainer(**trainer_dict, **test_model(),
                              **select_gpu_par(), precision=32,
                              log_every_n_steps=50, flush_logs_every_n_steps=500, max_epochs=30,
                              weights_summary='full', auto_lr_find=True)
        else:

            trainer = Trainer(**trainer_dict, max_epochs=500, **select_gpu_par(), log_every_n_steps=50,#limit_train_batches=0.1,
                              flush_logs_every_n_steps=500, resume_from_checkpoint=resume_from_checkpoint)
        trainer.fit(model, datamodule=data_module)
    # trainer.test(ckpt_path='best')
    else:
        anim = read_style_bvh("WildArms", "FW",[509,1009])

        check_file = ckpt_path + "/"

        modelfile = locate_model(check_file, args.epoch)

        model = DeepPhaseNet.load_from_checkpoint(modelfile)
        model = model.cuda()

        data_module.setup()

        app = Application(model, data_module)
        app = app.float()
        anim = subsample(anim, 1)
        app.setAnim(anim)
        app.forward()

        BVH.save_bvh("source.bvh",anim)


def readBVH(filename,dataset_property):
    remove_joints = (dataset_property['remove_joints'])
    if (remove_joints != None):
        remove_joints = remove_joints()
    filename = dataset_property["test_path"] + filename
    return BVH.read_bvh(filename, remove_joints=remove_joints, Tpose=-1, remove_gap=dataset_property['remove_gap'])

if __name__ == '__main__':
    setup_seed(3407)
    training_style100()


