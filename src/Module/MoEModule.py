import torch
from torch import nn
import math
class MultipleExpertsLinear(nn.Module):
    def __init__(self,in_channels,out_channels,nums):
        super(MultipleExpertsLinear, self).__init__()
        self.num_experts = nums
        weight_list = []
        bias_list = []
        for i in range(nums):
            weight = nn.Parameter(torch.empty(out_channels,in_channels))
            bias = nn.Parameter(torch.empty(out_channels))
            weight,bias = self.reset_parameter(weight,bias)
            weight_list.append(weight.unsqueeze(0))
            bias_list.append(bias.unsqueeze(0))
        self.weight = nn.Parameter(torch.cat(weight_list,dim=0))
        self.bias = nn.Parameter(torch.cat(bias_list,dim=0))
        self.in_channels = in_channels
        self.out_channels = out_channels

        #self.linear = torch.nn.Linear(in_channels,out_channels)
    def reset_parameter(self,weight,bias):
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(bias, -bound, bound)
        return weight,bias
    def forward(self,x,coefficients):
        '''coefficients: N,num_experts'''
        '''x: N,C'''
        mixed_weight = torch.einsum("bc,cmn->bmn",(coefficients,self.weight))

        mixed_bias = torch.matmul(coefficients,self.bias).unsqueeze(2)# N, out_channels, 1
        x = x.unsqueeze(2)  # N,1,in_channels
        out = torch.baddbmm(mixed_bias,mixed_weight,x).squeeze(2)
        return out


