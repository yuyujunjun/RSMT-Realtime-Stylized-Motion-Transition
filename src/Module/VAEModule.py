import torch
from torch import  nn
import torch.nn.functional as F
from src.geometry.quaternions import normalized_or6d
_EPS32 = torch.finfo(torch.float32).eps

class VAE_Conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super(VAE_Conv, self).__init__()
        assert kernel_size%2==1
        padding = kernel_size//2
        self.latent_size = out_channels
        self.encode = nn.Conv1d(in_channels,2*out_channels,kernel_size,padding=padding,padding_mode="reflect")
    def reparameterize(self,mu,log_sigma,epsilon = None):
        std = torch.exp(0.5*log_sigma)
        if epsilon==None:
            epsilon = torch.randn_like(std)
        return mu+std*epsilon
    def forward(self,input):
        x = self.encode(input)
        mu = x[:,:self.latent_size]
        log_var = x[:,self.latent_size:]
        return self.reparameterize(mu,log_var),mu,log_var
class VAE_Linear(nn.Module):
    def __init__(self,in_channels,out_channels,output_ori=True):
        super(VAE_Linear, self).__init__()
        self.output_ori = output_ori
        self.encode = nn.Linear(in_channels,2*out_channels)
        self.N = out_channels
        #self.encode_logvar = nn.Linear(in_channels,out_channels)
    def reparameterize(self,mu,log_sigma,epsilon = None):
        std = torch.exp(0.5*log_sigma)
        if epsilon==None:
            epsilon = torch.randn_like(std)
        return mu+std*epsilon
    def forward(self,input):
        mu_logvar = self.encode(input)
        mu,logvar = mu_logvar[:,:self.N],mu_logvar[:,self.N:]
        #logvar = self.encode_logvar(input)
        if(self.output_ori):
            return self.reparameterize(mu,logvar),mu,logvar
        else:
            return self.reparameterize(mu,logvar)
