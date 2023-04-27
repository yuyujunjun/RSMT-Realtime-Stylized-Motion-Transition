from torch import nn
import torch
class PLU(torch.nn.Module):
  def __init__(self,w=None,c=None):
    super(PLU, self).__init__()
    if w==None:
      self.w=torch.nn.Parameter(torch.Tensor([0.1,]))
    else:
      self.w=torch.nn.Parameter(torch.Tensor([w,]))
    self.w.requires_grad=True
    if(c == None):
      self.c=torch.nn.Parameter(torch.Tensor([1]))
    else:
      self.c=torch.nn.Parameter(torch.Tensor([c]))
    self.c.requires_grad=True

  def forward(self, x):
        # max(w(x+c)-c,min(w(x-c)+c,x))

        return torch.max(self.w * (x + self.c) - self.c, torch.min(self.w * (x - self.c) + self.c, x))
''' from [Liu et al.2019]'''










