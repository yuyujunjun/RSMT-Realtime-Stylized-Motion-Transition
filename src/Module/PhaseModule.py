import torch
from src.Module.Utilities import PLU

class PhaseSegement(torch.nn.Module):
    def __init__(self,phase_dims):
        super(PhaseSegement, self).__init__()
        self.phase_dims = phase_dims
        self.A_eps = torch.scalar_tensor(1e-6)

        self.F_act = torch.nn.Tanh()
    def forward(self,out):
        # x:[...,4*phase_dims]
        phase, A, F = out[..., :2 * self.phase_dims], out[..., 2 * self.phase_dims:3 * self.phase_dims], out[...,3 * self.phase_dims:4 * self.phase_dims]
        return phase.view(phase.shape[:-1]+(self.phase_dims,2)), A.unsqueeze(-1),F.unsqueeze(-1)
class PhaseOperator():
    def __init__(self,dt):
        self.dt = dt
        self._EPS32 = torch.finfo(torch.float32).eps
        self.tpi = 2 * torch.pi
        #self.phase_dim = phase_dim
        pass
    def phaseManifold(self,A,S):
        assert A.shape[-1] == 1
        out = torch.empty(size=(A.shape[:-1]+(2,)),device=A.device,dtype=A.dtype)
        out[...,0:1] = A*torch.cos(self.tpi * S)
        out[...,1:2] = A*torch.sin(self.tpi * S)
        return out
    def remove_F_discontiny(self,F):
        F = torch.where(F <= -0.5, 1 + F, F)
        F = torch.where(F >= 0.5, F - 1, F)
        return F
    def getA(self,phase):
        # phase should be (...,phase_dim,2)
        #phase = phase.view(phase.shape[:-1]+(self.phase_dim, 2))
        A = torch.norm(phase, dim=-1,keepdim=True)
        return A
    def normalized_phase(self,p):
        A0 = torch.norm(p, dim=-1, keepdim=True)
        A0 = torch.where(A0.abs() < self._EPS32, self._EPS32, A0)
        return p/A0

    def next_phase(self,last_phase,A,F):
        theta = self.tpi * F * self.dt
        #dA = self.dt * A
        #last_phase = last_phase.view(last_phase.shape[0], self.phase_dim, 2)
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        R = torch.stack([cos, sin, -sin, cos], dim=-1).reshape(cos.shape[:-1] + (2, 2))  # N,10,2,2

        # normalized
        baseA = torch.norm(last_phase,dim=-1,keepdim=True)
        baseA = torch.clamp_min(baseA,self._EPS32)
        last_phase = last_phase/baseA
        pred_phase = torch.matmul(last_phase.unsqueeze(dim=-2), R).squeeze(dim=-2)
        pred_phase = A*pred_phase
        # if (torch.isnan(last_phase).any() or torch.isinf(last_phase).any()):
        #     print("nan in last_phase")
        # elif (torch.isnan(R).any()):
        #     print("nan in R")
        # elif(torch.isnan(pred_phase).any()):
        #     print("nan in pred_phase")

        #pred_phase = pred_phase/baseA
        #pred_phase = pred_phase * (dA+baseA)
        return pred_phase

    def slerp(self,p0,p1):

        # only work for t = 0.5, if p0 \approx -p1, the results might have numerical error, we assume p0 is similar to p1
        A0 = torch.norm(p0,dim=-1,keepdim=True)
        A1 = torch.norm(p1,dim=-1,keepdim=True)
        A0 = torch.where(A0.abs()<self._EPS32,self._EPS32,A0)
        A1 = torch.where(A1.abs()<self._EPS32,self._EPS32,A1)
        # normalized
        p0 = p0/A0
        p1 = p1/A1
        p = self.normalized_phase(0.5*(p0+p1))

        A = 0.5*(A0+A1)
        return A*p