import copy
import os
import re
from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.utilities.seed import seed_everything
from src.Datasets.BaseLoader import WindowBasedLoader
from src.Datasets.Style100Processor import StyleLoader, Swap100StyJoints
from src.utils import BVH_mod as BVH
from src.utils.motion_process import subsample


def setup_seed(seed:int):
    seed_everything(seed,True)
def test_model():
    dict = {}
    #dict['fast_dev_run'] = 1 # only run 1 train, val, test batch and program ends
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
    parser.add_argument("--epoch",type=str,default="last")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test",action="store_true")
    parser.add_argument("--moe_model",type=str,default="./results/StyleVAE2_style100/myResults/55/m_save_model_332")
    parser.add_argument("--pretrained",action="store_true")
    parser.add_argument("--predict_phase",action="store_true")
    args = parser.parse_args()
    ckpt_path_prefix = "results/"
    if (args.version != "-1"):
        version = args.version
    else:
        version = None
    '''Create Loggers tensorboard'''
    if args.dev_run:
        log_name += "dev_run"
    else:
        log_name += "myResults"
    tb_logger = pl.loggers.TensorBoardLogger(save_dir="tensorboard_logs/", name=log_name, version=None)
    #load_ckpt_path = os.path.join(ckpt_path_prefix, prefix+'/myResults', str(version))
    load_ckpt_path = os.path.join(ckpt_path_prefix, prefix+'/myResults', str(version))
    save_ckpt_path = os.path.join(ckpt_path_prefix, log_name, str(tb_logger.version))

    if (args.resume == True):
        check_file = load_ckpt_path+"/"
        if (args.epoch == "last"):
            check_file += "last.ckpt"
        else:
            dirs = os.listdir(check_file)
            for dir in dirs:
                st = "epoch=" + args.epoch + "-step=\d+.ckpt"
                out = re.findall(st, dir)
                if (len(out) > 0):
                    check_file += out[0]
                    print(check_file)
                    break
        resume_from_checkpoint = check_file  # results/version/last.ckpt"
    else:
        resume_from_checkpoint = None
    checkpoint_callback = [ModelCheckpoint(dirpath=save_ckpt_path + "/", save_top_k=-1, save_last=False, every_n_epochs=2,save_weights_only=True),
                           ModelCheckpoint(dirpath=save_ckpt_path + "/", save_top_k=1, monitor="val_loss", save_last=True, every_n_epochs=1,save_weights_only=True),
                          # EMA(0.99)
                           ]
    '''Train'''
    checkpoint_callback[0].CHECKPOINT_NAME_LAST = "last"
    profiler = SimpleProfiler()#PyTorchProfiler(filename="profiler")
    trainer_dict = {
        "callbacks":checkpoint_callback,
        "profiler":profiler,
        "logger":tb_logger
    }
    return args,trainer_dict,load_ckpt_path
def read_style_bvh(style,content,clip=None):
    swap_joints = Swap100StyJoints()
    anim = BVH.read_bvh(os.path.join("MotionData/100STYLE/",style,style+"_"+content+".bvh"),remove_joints=swap_joints)
    if (clip != None):
        anim.quats = anim.quats[clip[0]:clip[1], ...]
        anim.hip_pos = anim.hip_pos[clip[0]:clip[1], ...]
    anim = subsample(anim,ratio=2)
    return anim

def training_style100_phase():
    from src.Datasets.StyleVAE_DataModule import StyleVAE_DataModule
    from src.Net.TransitionPhaseNet import TransitionNet_phase,Application_phase
    prefix = "Transitionv2"
    data_set = "style100"
    prefix += "_" + data_set
    args, trainer_dict, ckpt_path = create_common_states(prefix)
    moe_net = torch.load(args.moe_model)

    if(args.pretrained==True):
        from src.utils.locate_model import locate_model
        pretrained_file = locate_model(ckpt_path+"/",args.epoch)
        pre_trained = torch.load(pretrained_file)
    else:
        pre_trained = None



    loader = WindowBasedLoader(61, 21, 1)
    dt = 1. / 30.
    phase_dim = 10
    phase_file = "+phase_gv10"
    style_file_name = phase_file + WindowBasedLoader(120,0,1).get_postfix_str()
    if (args.test == False):
        '''Create the model'''
        style_loader = StyleLoader()

        data_module = StyleVAE_DataModule(style_loader, phase_file + loader.get_postfix_str(),style_file_name, dt=dt,
                                         batch_size=32,mirror=0.0) # when apply phase, should avoid mirror
        stat = style_loader.load_part_to_binary("motion_statistics")
        mode = "pretrain"

        model = TransitionNet_phase(moe_net, data_module.skeleton, pose_channels=9,stat=stat ,phase_dim=phase_dim,
                               dt=dt,mode=mode,pretrained_model=pre_trained,predict_phase=args.predict_phase)

        if (args.dev_run):
            trainer = Trainer(**trainer_dict, **test_model(),
                              **select_gpu_par(), precision=32,reload_dataloaders_every_n_epochs=1,
                              log_every_n_steps=5, flush_logs_every_n_steps=10,
                              weights_summary='full')
        else:

            trainer = Trainer(**trainer_dict, max_epochs=10000,reload_dataloaders_every_n_epochs=1,gradient_clip_val=1.0,
                              **select_gpu_par(), log_every_n_steps=50,check_val_every_n_epoch=2,
                              flush_logs_every_n_steps=100)
        trainer.fit(model, datamodule=data_module)
    else:

        style_loader = StyleLoader()
        data_module = StyleVAE_DataModule(style_loader, phase_file + loader.get_postfix_str(),style_file_name, dt=dt,batch_size=32,mirror=0.0)
        data_module.setup()

        check_file = ckpt_path + "/"
        if (args.epoch == "last"):
            check_file += "last.ckpt"
            print(check_file)
        else:
            dirs = os.listdir(check_file)
            for dir in dirs:
                st = "epoch=" + args.epoch + "-step=\d+.ckpt"
                out = re.findall(st, dir)
                if (len(out) > 0):
                    check_file += out[0]
                    print(check_file)
                    break
        model = TransitionNet_phase.load_from_checkpoint(check_file, moe_decoder=moe_net, pose_channels=9,phase_dim=phase_dim,
                               dt=dt,mode='fine_tune',strict=False)
        model = model.cuda()
        data_module.mirror = 0
        app = Application_phase(model, data_module)
        model.eval()

        app = app.float()

        key = "HighKnees"
        sty_key = "HighKnees"
        cid = 61
        sid = 4
        src_motion = app.data_module.test_set.dataset[key][cid]
        target_motion = app.data_module.test_set_sty.dataset[sty_key][sid]

        app.setSource(src_motion)
        app.setTarget(target_motion)
        source = BVH.read_bvh("source.bvh")
        output = copy.deepcopy(source)

        output.hip_pos, output.quats = app.forward(t=2., x=0.)
        BVH.save_bvh("test_net.bvh", output)
        output.hip_pos, output.quats = app.get_source()
        BVH.save_bvh("source.bvh", output)
        torch.save(model, ckpt_path + "/m_save_model_" + str(args.epoch))

if __name__ == '__main__':
    setup_seed(3407)
    training_style100_phase()


