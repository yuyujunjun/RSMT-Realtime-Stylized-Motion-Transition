from torch import nn
from torch._lowrank import pca_lowrank
import pytorch_lightning as pl
from src.Net.CommonOperation import CommonOperator
from src.utils.Drawer import Drawer
from src.Datasets.DeepPhaseDataModule import DeephaseDataSet
import numpy as np
from torch.optim.optimizer import Optimizer


class AdamW(Optimizer):
    """Implements Adam algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        #super(AdamW, self).__init__(params, defaults)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.mul_(1 - group['weight_decay']).addcdiv_(exp_avg, denom, value=-step_size)

        return loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
import torch


class ReduceMaxLROnRestart:
    def __init__(self, ratio=0.75):
        self.ratio = ratio

    def __call__(self, eta_min, eta_max):
        return eta_min, eta_max * self.ratio


class ExpReduceMaxLROnIteration:
    def __init__(self, gamma=1):
        self.gamma = gamma

    def __call__(self, eta_min, eta_max, iterations):
        return eta_min, eta_max * self.gamma ** iterations


class CosinePolicy:
    def __call__(self, t_cur, restart_period):
        return 0.5 * (1. + math.cos(math.pi *
                                    (t_cur / restart_period)))


class ArccosinePolicy:
    def __call__(self, t_cur, restart_period):
        return (math.acos(max(-1, min(1, 2 * t_cur
                                      / restart_period - 1))) / math.pi)


class TriangularPolicy:
    def __init__(self, triangular_step=0.5):
        self.triangular_step = triangular_step

    def __call__(self, t_cur, restart_period):
        inflection_point = self.triangular_step * restart_period
        point_of_triangle = (t_cur / inflection_point
                             if t_cur < inflection_point
                             else 1.0 - (t_cur - inflection_point)
                             / (restart_period - inflection_point))
        return point_of_triangle


class CyclicLRWithRestarts(_LRScheduler):
    """Decays learning rate with cosine annealing, normalizes weight decay
    hyperparameter value, implements restarts.
    https://arxiv.org/abs/1711.05101

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        batch_size: minibatch size
        epoch_size: training samples per epoch
        restart_period: epoch count in the first restart period
        t_mult: multiplication factor by which the next restart period will expand/shrink
        policy: ["cosine", "arccosine", "triangular", "triangular2", "exp_range"]
        min_lr: minimum allowed learning rate
        verbose: print a message on every restart
        gamma: exponent used in "exp_range" policy
        eta_on_restart_cb: callback executed on every restart, adjusts max or min lr
        eta_on_iteration_cb: callback executed on every iteration, adjusts max or min lr
        triangular_step: adjusts ratio of increasing/decreasing phases for triangular policy


    Example:
        >>> scheduler = CyclicLRWithRestarts(optimizer, 32, 1024, restart_period=5, t_mult=1.2)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>         ...
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>         scheduler.batch_step()
        >>>     validate(...)
    """

    def __init__(self, optimizer, batch_size, epoch_size, restart_period=100,
                 t_mult=2, last_epoch=-1, verbose=False,
                 policy="cosine", policy_fn=None, min_lr=1e-7,
                 eta_on_restart_cb=None, eta_on_iteration_cb=None,
                 gamma=1.0, triangular_step=0.5):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        self.optimizer = optimizer

        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
                group.setdefault('minimum_lr', min_lr)
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an"
                                   " optimizer".format(i))

        self.base_lrs = [group['initial_lr'] for group
                         in optimizer.param_groups]

        self.min_lrs = [group['minimum_lr'] for group
                        in optimizer.param_groups]

        self.base_weight_decays = [group['weight_decay'] for group
                                   in optimizer.param_groups]

        self.policy = policy
        self.eta_on_restart_cb = eta_on_restart_cb
        self.eta_on_iteration_cb = eta_on_iteration_cb
        if policy_fn is not None:
            self.policy_fn = policy_fn
        elif self.policy == "cosine":
            self.policy_fn = CosinePolicy()
        elif self.policy == "arccosine":
            self.policy_fn = ArccosinePolicy()
        elif self.policy == "triangular":
            self.policy_fn = TriangularPolicy(triangular_step=triangular_step)
        elif self.policy == "triangular2":
            self.policy_fn = TriangularPolicy(triangular_step=triangular_step)
            self.eta_on_restart_cb = ReduceMaxLROnRestart(ratio=0.5)
        elif self.policy == "exp_range":
            self.policy_fn = TriangularPolicy(triangular_step=triangular_step)
            self.eta_on_iteration_cb = ExpReduceMaxLROnIteration(gamma=gamma)

        self.last_epoch = last_epoch
        self.batch_size = batch_size
        self.epoch_size = epoch_size

        self.iteration = 0
        self.total_iterations = 0

        self.t_mult = t_mult
        self.verbose = verbose
        self.restart_period = math.ceil(restart_period)
        self.restarts = 0
        self.t_epoch = -1
        self.epoch = -1

        self.eta_min = 0
        self.eta_max = 1

        self.end_of_period = False
        self.batch_increments = []
        self._set_batch_increment()

    def _on_restart(self):
        if self.eta_on_restart_cb is not None:
            self.eta_min, self.eta_max = self.eta_on_restart_cb(self.eta_min,
                                                                self.eta_max)

    def _on_iteration(self):
        if self.eta_on_iteration_cb is not None:
            self.eta_min, self.eta_max = self.eta_on_iteration_cb(self.eta_min,
                                                                  self.eta_max,
                                                                  self.total_iterations)

    def get_lr(self, t_cur):
        eta_t = (self.eta_min + (self.eta_max - self.eta_min)
                 * self.policy_fn(t_cur, self.restart_period))

        weight_decay_norm_multi = math.sqrt(self.batch_size /
                                            (self.epoch_size *
                                             self.restart_period))

        lrs = [min_lr + (base_lr - min_lr) * eta_t for base_lr, min_lr
               in zip(self.base_lrs, self.min_lrs)]
        weight_decays = [base_weight_decay * eta_t * weight_decay_norm_multi
                         for base_weight_decay in self.base_weight_decays]

        if (self.t_epoch + 1) % self.restart_period < self.t_epoch:
            self.end_of_period = True

        if self.t_epoch % self.restart_period < self.t_epoch:
            if self.verbose:
                print("Restart {} at epoch {}".format(self.restarts + 1,
                                                      self.last_epoch))
            self.restart_period = math.ceil(self.restart_period * self.t_mult)
            self.restarts += 1
            self.t_epoch = 0
            self._on_restart()
            self.end_of_period = False

        return zip(lrs, weight_decays)

    def _set_batch_increment(self):
        d, r = divmod(self.epoch_size, self.batch_size)
        batches_in_epoch = d + 2 if r > 0 else d + 1
        self.iteration = 0
        self.batch_increments = torch.linspace(0, 1, batches_in_epoch).tolist()

    def step(self):
        self.last_epoch += 1
        self.t_epoch += 1
        self._set_batch_increment()
        self.batch_step()

    def batch_step(self):
        try:
            t_cur = self.t_epoch + self.batch_increments[self.iteration]
            self._on_iteration()
            self.iteration += 1
            self.total_iterations += 1
        except (IndexError):
            raise StopIteration("Epoch size and batch size used in the "
                                "training loop and while initializing "
                                "scheduler should be the same.")

        for param_group, (lr, weight_decay) in zip(self.optimizer.param_groups,
                                                   self.get_lr(t_cur)):
            param_group['lr'] = lr
            param_group['weight_decay'] = weight_decay
        return lr

def PhaseManifold(A,S):
    shape = list(A.shape)
    shape[-1]*=2
    output = torch.empty((shape))
    output[...,::2] = A*torch.cos(2*torch.pi*S)
    output[...,1::2] = A*torch.sin(2*torch.pi*S)
    return output



class PAE_AI4Animation(nn.Module):
    def __init__(self,n_phases, n_joints,length, key_range=1., window=2.0):
        super(PAE_AI4Animation, self).__init__()
        embedding_channels = n_phases
        input_channels = (n_joints)*3
        time_range = length
        self.n_phases = n_phases
        self.input_channels = input_channels
        self.embedding_channels = embedding_channels
        self.time_range = time_range
        self.key_range = key_range

        self.window = window
        self.time_scale = key_range / time_range

        self.tpi = nn.Parameter(torch.from_numpy(np.array([2.0 * np.pi], dtype=np.float32)), requires_grad=False)
        self.args = nn.Parameter(
            torch.from_numpy(np.linspace(-self.window / 2, self.window / 2, self.time_range, dtype=np.float32)),
            requires_grad=False)
        self.freqs = nn.Parameter(torch.fft.rfftfreq(time_range)[1:] * (time_range * self.time_scale) / self.window,
                               requires_grad=False)  # Remove DC frequency

        intermediate_channels = int(input_channels / 3)

        self.conv1 = nn.Conv1d(input_channels, intermediate_channels, time_range, stride=1,
                               padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.bn_conv1 = nn.BatchNorm1d(num_features=intermediate_channels)
        self.conv2 = nn.Conv1d(intermediate_channels, embedding_channels, time_range, stride=1,
                               padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.bn_conv2 = nn.BatchNorm1d(num_features=embedding_channels)

        self.fc = torch.nn.ModuleList()
        self.bn = torch.nn.ModuleList()
        for i in range(embedding_channels):
            self.fc.append(nn.Linear(time_range, 2))
            self.bn.append(nn.BatchNorm1d(num_features=2))
        self.parallel_fc0 = nn.Linear(time_range,embedding_channels)
        self.parallel_fc1 = nn.Linear(time_range,embedding_channels)

        self.deconv1 = nn.Conv1d(embedding_channels, intermediate_channels, time_range, stride=1,
                                 padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True,
                                 padding_mode='zeros')
        self.bn_deconv1 = nn.BatchNorm1d(num_features=intermediate_channels)
        self.deconv2 = nn.Conv1d(intermediate_channels, input_channels, time_range, stride=1,
                                 padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True,
                                 padding_mode='zeros')

    def atan2(self, y, x):
        tpi = self.tpi
        ans = torch.atan(y / x)
        ans = torch.where((x < 0) * (y >= 0), ans + 0.5 * tpi, ans)
        ans = torch.where((x < 0) * (y < 0), ans - 0.5 * tpi, ans)
        return ans

    # Returns the frequency for a function over a time window in s
    def FFT(self, function, dim):
        rfft = torch.fft.rfft(function, dim=dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:, :, 1:]  # Spectrum without DC component
        power = spectrum ** 2

        # Frequency
        freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)
        freq = freq / self.time_scale

        # Amplitude
        amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.time_range

        # Offset
        offset = rfft.real[:, :, 0] / self.time_range  # DC component

        return freq, amp, offset

    def forward(self, x):
        y = x

        # Signal Embedding
        y = y.reshape(y.shape[0], self.input_channels, self.time_range)

        y = self.conv1(y)
        y = self.bn_conv1(y)
        y = torch.tanh(y)

        y = self.conv2(y)
        y = self.bn_conv2(y)
        y = torch.tanh(y)

        latent = y  # Save latent for returning

        # Frequency, Amplitude, Offset
        f, a, b = self.FFT(y, dim=2)

        # Phase
        sx = self.parallel_fc0(y).diagonal(dim1=-2,dim2=-1).unsqueeze(-1).contiguous()
        sy = self.parallel_fc1(y).diagonal(dim1=-2,dim2=-1).unsqueeze(-1).contiguous()
        v = torch.cat([sx,sy],dim=-1) # B x M x 2
        tv = torch.empty_like(v)
        for i in range(self.embedding_channels):
            tv[:,i,:] = self.bn[i](v[:,i,:])
        p = self.atan2(tv[:,:,1],tv[:,:,0])/self.tpi

        #################### the original code ####################
        # p = torch.empty((y.shape[0], self.embedding_channels), dtype=torch.float32, device=y.device)
        # for i in range(self.embedding_channels):
        #     v = self.fc[i](y[:, i, :])
        #     v = self.bn[i](v)
        #     p[:, i] = self.atan2(v[:, 1], v[:, 0]) / self.tpi
        ###########################################################


        # Parameters
        p = p.unsqueeze(2)
        f = f.unsqueeze(2)
        a = a.unsqueeze(2)
        b = b.unsqueeze(2)
        params = [p, f, a, b]  # Save parameters for returning

        # Latent Reconstruction
        y = a * torch.sin(self.tpi * (f * self.args + p)) + b

        signal = y  # Save signal for returning

        # Signal Reconstruction
        y = self.deconv1(y)
        y = self.bn_deconv1(y)
        y = torch.tanh(y)

        y = self.deconv2(y)

        return y, p, a, f, b





class DeepPhaseNet(pl.LightningModule):
    def __init__(self,n_phase,skeleton,length,dt,batch_size):
        super(DeepPhaseNet, self).__init__()
       # self.automatic_optimization = False
        self.save_hyperparameters(ignore=['style_loader'])
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.dt = dt
        self.skeleton  = skeleton
        self.model = PAE_AI4Animation(n_phase,skeleton.num_joints,length)
        self.mse_loss = nn.MSELoss()
        self.oper = CommonOperator(batch_size)
    def transform_to_pca(self,input):
        # input: N,T,J,D
        input = input*self.dt
        std = torch.std(input,dim=[-2,-1],keepdim=True)
        mean = torch.mean(input,dim=[-2,-1],keepdim=True)
        input = (input-mean)/std
        input = input.flatten(1,2)
        return input
    def forward(self,input):
        input = input.flatten(1,2)
        Y,S,A,F,B = self.model(input)
        loss ={ "loss":self.mse_loss(input,Y)}
        return loss,input,Y
    def training_step(self, batch,batch_idx):
        loss,_,_ = self.forward(batch)
        self.oper.log_dict(self,loss,"train_")
        return loss['loss']
    def validation_step(self, batch,batch_idx) :
        loss,input,Y = self.forward(batch)
        self.oper.log_dict(self, loss, "val_")
        return loss['loss']
    def test_step(self,batch,batch_idx):
        loss = self.forward(batch)
        self.oper.log_dict(self,loss,"test_")
        return loss['loss']
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CyclicLRWithRestarts(optimizer=optimizer, batch_size=32,
                                          epoch_size=75830, restart_period=10,
                                          t_mult=2, policy="cosine", verbose=True)
        return [optimizer], {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
'''增加一个loss计算，确保输入和网络训练的时候是一致的'''
from src.Datasets.DeepPhaseDataModule import DeepPhaseProcessor
class Application(nn.Module):
    def __init__(self,Net:DeepPhaseNet,datamodule):
        super(Application, self).__init__()
        self.Net = Net
        self.drawer = Drawer()
        self.module = datamodule
        self.processor = DeepPhaseProcessor (1/30.)
        self.window = datamodule.window
    def draw_phase(self,label,input,channel,feature):
        '''input: Batch x M x 1'''
        self.drawer.ax[channel,feature].set_xlabel(label)
        self.drawer.draw(input.cpu().numpy(),channel=channel,feature=feature)
    def draw_dict(self,input:dict):
        size = len(input)
        print(self.Net.model.n_phases)
        self.drawer.create_canvas(self.Net.model.n_phases,size)
        feature_id = 0
        for key in input.keys():
            data = input[key]
            for phase in range(data.shape[1]):
                self.draw_phase(key,data[:,phase],phase,feature_id)
            feature_id+=1

    def _transform_anim(self,anim):
        offsets,hip_pos,quats = anim.offsets,anim.hip_pos,anim.quats#self.loader(anim)
        offsets = np.expand_dims(offsets,axis=0)
        hip_pos = np.expand_dims(hip_pos,axis=0)
        quats = np.expand_dims(quats,axis=0)
        gv = self.processor.transform_single(offsets,hip_pos,quats,self.module.skeleton)
        batch = gv
        return batch
    def setAnim(self,anim):
        self.input = self._transform_anim(anim).to(torch.float32)#.cpu()
        self.dataset = DeephaseDataSet(self.input,self.window)
    def forward_window(self,batch):
        input = batch
        input = self.module.on_after_batch_transfer(input,0)
        return self.calculate_phase(input)
    def calculate_FFT(self,embedding):
        N = embedding.shape[-1]
        model = self.Net.model
        A, F, B = model.fft_layer(embedding, N)
        return A,F,B
    def calculate_phase(self,x):
        N = x.shape[-1]
        model = self.Net.model
        self.Net = self.Net.to(self.Net.device)
        x = x.to(self.Net.device)
        self.Net.eval()
        x = x.flatten(1,2 )
        Y,S,A,F,B = model.forward(x)
        print("mseLoss:{}".format(self.Net.mse_loss(Y,x)))
        return {
            "S":S,"A":A,"F":F,"B":B
        }
    def calculate_statistic_for_dataset(self,dataset):
        self.Net.eval()
        with torch.no_grad():
            phases = {"F": [], "A": [], "S": [], "B": []}
            step = 1
            length = len(dataset)
            input = []
            for i in range(length):
                input.append(dataset[i].unsqueeze(0))

            batch = torch.cat(input, 0)
            batch = batch.to("cuda")
            phase = self.forward_window(batch)
            for key in phase.keys():
                phase[key] = phase[key].cpu().numpy()
            return phase
    def forward(self):
        import matplotlib.pyplot as plt
        self.Net.eval()
        with torch.no_grad():
            start = 0
            length = max(self.window,len(self.dataset)-self.window-10)
            input = []
            for i in range(length):
                input.append(self.dataset[i+start].unsqueeze(0))

            batch = torch.cat(input,0)
            phase = self.forward_window(batch)
            self.draw_dict(phase)
            self.drawer.show()

            phase = PhaseManifold(phase['A'][:,:,0],phase['S'][:,:,0])
            U,S,V = pca_lowrank(phase)
            proj = torch.matmul(phase,V)
            proj = proj[:,:2]
            c = np.arange(0,proj.shape[0],step=1.)
            plt.scatter(proj[:,0],proj[:,1],c=c)
            plt.show()



