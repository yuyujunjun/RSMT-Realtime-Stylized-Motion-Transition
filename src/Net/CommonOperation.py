from torch import nn
import pytorch_lightning as pl
import torch
import math
class CommonOperator():
    def __init__(self,batch_size):
        self.batch_size = batch_size
        self.steps_per_epoch = None
        pass

    def add_prefix_to_loss(self, loss:dict, prefix:str):
        output = {}
        for key, value in loss.items():
            if(type(value)==torch.Tensor):
                output[prefix + key] = value.detach()
            else:
                output[prefix + key] = value
        return output
    def set_lr(self,lr,optimizer):
        for pg in optimizer.param_groups:
            pg['lr'] = lr
    def log_dict(self,obj:pl.LightningModule,loss:dict,prefix:str):
        loss_ = self.add_prefix_to_loss(loss,prefix)
        obj.log_dict(loss_,prog_bar=True,logger=True,batch_size=self.batch_size)
        return loss_

    def add_weight_decay(self,model,lr, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': no_decay, 'weight_decay': 0.,"lr":lr},
            {'params': decay, 'weight_decay': weight_decay,"lr":lr}]
    def num_training_steps(self,obj:pl.LightningModule) -> int:
        """Total training steps inferred from datamodule and devices."""
        # warning: will call train_dataloader()
        if(self.steps_per_epoch==None):
            dataset = obj.trainer._data_connector._train_dataloader_source.dataloader()
            self.steps_per_epoch=len(dataset)//obj.trainer.accumulate_grad_batches
        return self.steps_per_epoch

    def get_progress(self,obj:pl.LightningModule,target_epoch:float,start_epoch=0.):
        steps_per_epoch = self.num_training_steps(obj)
        return max(min((obj.global_step-start_epoch*steps_per_epoch)/(target_epoch*steps_per_epoch-start_epoch*steps_per_epoch),1.0),0.0)
    def collect_models(self,obj:nn.Module):
        models = []
        for i, module in obj._modules.items():
            if (sum(p.numel() for p in module.parameters()) > 0):
                models.append(i)
        return models

