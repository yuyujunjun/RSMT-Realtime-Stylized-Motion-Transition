import torch
import numpy as np
_EPS64 = torch.finfo(torch.float64).eps
_4EPS64 = _EPS64 * 4.0
_EPS32 = torch.finfo(torch.float32).eps
_4EPS32 = _EPS32 * 4.0
# batch*n
def find_secondary_axis(v):
    v = normalize_vector(v)
    refx = torch.tensor(np.array([[1, 0, 0]]), dtype=v.dtype, device=v.device).expand(v.shape)
    refy =  torch.tensor(np.array([[0, 1, 0]]), dtype=v.dtype, device=v.device).expand(v.shape)
    dot = lambda x,y:(x*y).sum(-1,keepdim=True)
    refxv,refyv = dot(v,refx).abs(),dot(v,refy).abs()    # find proper axis
    ref = torch.where(refxv>refyv,refy,refx)
    pred = torch.cross(v,ref,dim=-1)
    pred = normalize_vector(pred)
    return pred

def normalize_vector(v, return_mag=False):
    if(v.dtype==torch.float64):
        eps = _EPS64
    else:
        eps = _EPS32
    v_mag = torch.norm(v,dim=-1,keepdim=True)# torch.sqrt(v.pow(2).sum(-1,keepdim=True))#+eps

    #v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([eps]).type_as(v)))
    v = v / v_mag
    if return_mag:
        return v, v_mag[:, 0]
    else:
        return v


def cross_product(u, v):
    """
    Cross operation on batched vectors of shape (..., 3)
    """
    i = u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1]
    j = u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2]
    k = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    out = torch.cat((i.unsqueeze(-1), j.unsqueeze(-1), k.unsqueeze(-1)), dim=-1)
    return out
