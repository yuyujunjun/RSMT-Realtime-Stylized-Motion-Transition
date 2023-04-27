import math
import random

import torch
from torch import nn


class PosEncoding(nn.Module):
    def positional_encoding(self,tta, basis=10000., dimensions=256):
        z = torch.zeros(dimensions, device='cuda')  # .type(torch.float16)
        indices = torch.arange(0, dimensions, 2, device='cuda').type(torch.float32)
        z[indices.long()] = torch.sin(tta / torch.pow(basis, indices / dimensions))
        z[indices.long() + 1] = torch.cos(tta / torch.pow(basis, indices / dimensions))
        return z
    def __init__(self,max_tta,dims):
        super(PosEncoding, self).__init__()
        ttaEncoding = torch.empty(max_tta, dims)
        for tta in range(max_tta):
            ttaEncoding[tta] = self.positional_encoding(tta,basis=10000.,dimensions=dims)
        self.register_buffer("ttaEncoding", ttaEncoding)
    def forward(self,id):
        return self.ttaEncoding[id]
def add_pos_info(latent,embedings,tta,t_max):
    t = min(tta, t_max)
    pos = embedings(t)
    #pos = torch.cat((pos,pos,pos),dim=-1)
    return latent+pos

def multi_concat(context_state,context_offset,context_target,embeddings,embeddings512,noise_per_sequence,tta,t_max):
    t = torch.where(tta < t_max, tta, t_max)
    embeddings_vector = []
    embeddings512_vector = []
    for i in range(t.shape[0]):
        embeddings_vector.append(embeddings(t[i]).view(1,1,-1))
        embeddings512_vector.append(embeddings512(t[i]).view(1,1,-1))
    embeddings_vector = torch.cat(embeddings_vector,dim=1) # 1, 8 ,256
    embeddings512_vector = torch.cat(embeddings512_vector,dim=1)
    h_target = context_target + embeddings_vector
    h_state = context_state + embeddings512_vector
    h_offset = context_offset + embeddings_vector

    lambda_tar = torch.where(t>30,1.,torch.where(t<5.,0,(t-5.)/25.))
    # if tta >= 30:
    #     lambda_tar = 1.
    # elif tta < 5:
    #     lambda_tar = 0.
    # else:
    #     lambda_tar = (tta - 5.) / 25.
    h_target = torch.cat((h_offset, h_target), dim=-1)
    h_target = h_target + lambda_tar.view(1,-1,1) * noise_per_sequence.unsqueeze(1)

    return torch.cat((h_state, h_target), dim=-1), h_target
def concat(context_state,context_offset,context_target,embeddings,embeddings512,noise_per_sequence,tta,t_max):
#    t = torch.min(tta,t_max) # MAXFRAME+10-5
    t = min(tta,t_max)
    h_target = context_target + embeddings(t)
    h_state = context_state + embeddings512(t)
    h_offset = context_offset + embeddings(t)
    if tta >= 30:
        lambda_tar = 1.
    elif tta < 5:
        lambda_tar = 0.
    else:
        lambda_tar = (tta - 5.) / 25.
    h_target = torch.cat((h_offset, h_target), dim=-1)
    h_target = h_target + lambda_tar * noise_per_sequence

    return torch.cat((h_state,h_target),dim=-1),h_target

class SeqScheduler():
    def __init__(self,initial_seq,max_seq):
        self.initial_seq = initial_seq
        self.max_seq = max_seq
       # self.epoch = epoch
    def progress(self,t:float):
        t = min(max(t,0.),1.)
        out = (self.max_seq-self.initial_seq)*t+self.initial_seq
        return int(out)
    def range(self,t:float):
        upper = self.progress(t)
        return random.randint(self.initial_seq,upper)

class ATNBlock(nn.Module):
    def __init__(self,content_dims,style_dims):
        super(ATNBlock, self).__init__()
        in_ch = content_dims
        self.in_ch = content_dims
        self.sty_ch = style_dims
        self.f = nn.Linear(content_dims, style_dims)
        self.g = nn.Linear(style_dims, style_dims)
        self.h = nn.Linear(style_dims, style_dims)
        self.sm = nn.Softmax(dim=-2)
        self.k = nn.Linear(style_dims, in_ch)
        self.norm = nn.InstanceNorm2d(style_dims,affine=False)
        self.norm_content = nn.InstanceNorm1d(content_dims,affine=False) #LayerNorm(keep the same as AdaIn)
       # self.s = []
    def forward(self, fs, fd,pos, first):
        #return fd
        N,T,C = fs.shape
        N,C = fd.shape
        x = fd
        s_sty = fs
        # N,C : fd.shape
        # N,C,T : fs.shape
        b = s_sty.shape[0]

        F = self.f(self.norm_content(x)).unsqueeze(-1) # N,C,1
        if(first):
            G = self.g(self.norm(s_sty)) #N,T,C
            self.G = G.view(b, -1, self.sty_ch)  # N,T,C
            #s_sty_pos = s_sty+pos_embedding.view(1,C,1)
            self.H = self.h(s_sty).transpose(1,2) #N,C,T
        #H = H.view(b, self.sty_ch, -1)  # N,C,T


        F = F.view(b, self.sty_ch, -1) #N,C,1

        S = torch.bmm(self.G,F) #N,T,1

        S = self.sm(S/math.sqrt(self.G.shape[-1]))
        # self.s.append(S)
        O = torch.bmm(self.H, S) # N,C,1

        O = O.view(x.shape[:-1]+(self.sty_ch,))

        O = self.k(O)
        O += x
        return O

class AdaInNorm2D(nn.Module):
    r"""MLP(fs.mean(),fs.std()) -> instanceNorm(fd)"""
    def __init__(self,style_dims,content_dim,n_joints=0):

        super(AdaInNorm2D, self).__init__()

        self.affine2 = nn.Linear(style_dims, style_dims)
        self.act = nn.ELU()#nn.LeakyReLU(0.2)
        self.affine3 = nn.Linear(style_dims, style_dims)
        self.affine4 = nn.Linear(style_dims, content_dim * 2)
        self.norm = nn.InstanceNorm1d(512)
        self.dropout = nn.Dropout(0.1)

    def forward(self, s, d ,pos_emedding,first):
        if(first):
            N,T,C = s.shape
            s = torch.mean(s,dim=1) #N,C
            s = self.affine2(s)
            s = self.act(s)
            s = self.dropout(s)
            s = self.affine4(s)
            self.gamma, self.beta = torch.chunk(s, chunks=2, dim=1)

        d = self.norm(d)
        return (1 + self.gamma) * d + self.beta
