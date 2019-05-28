#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
class Highway(nn.Module):

    def __init__(self,emb_size):
        super(Highway,self).__init__()
        self.gate = nn.Sequential(nn.Linear(emb_size,emb_size,bias=True),nn.Sigmoid())
        self.proj = nn.Sequential(nn.Linear(emb_size,emb_size,bias=True),nn.ReLU())
        # self.gate_linear = nn.Linear(emb_size,emb_size,bias=True)
        # self.proj_linear = nn.Linear(emb_size,emb_size,bias=True)
        # print(self.gate_linear.weight)
        # for child in self.gate.children():
        #     print(child.weight)
        '''
        测试随机种子影响-与weight出现顺序有关
        
        self.gate = nn.Sequential(nn.Linear(emb_size,emb_size,bias=True),nn.Sigmoid())
        self.proj = nn.Sequential(nn.Linear(emb_size,emb_size,bias=True),nn.ReLU())
        self.gate_linear = nn.Linear(emb_size,emb_size,bias=True)
        self.proj_linear = nn.Linear(emb_size,emb_size,bias=True)
        print(self.gate_linear.weight)
        for child in self.gate.children():
             print(child.weight)
        
        Parameter containing:
        tensor([[ 0.5387,  0.2771, -0.0558],
                [-0.0280,  0.3282, -0.4013],
                [ 0.1919, -0.1913,  0.3340]], requires_grad=True)
        Parameter containing:
        tensor([[-0.5439, -0.1133, -0.2773],
                [-0.1540, -0.5100,  0.2317],
                [-0.5175, -0.0368,  0.2007]], requires_grad=True)

        
        self.gate_linear = nn.Linear(emb_size,emb_size,bias=True)
        self.proj_linear = nn.Linear(emb_size,emb_size,bias=True)
        print(self.gate_linear.weight)
        Parameter containing:
            tensor([[-0.5439, -0.1133, -0.2773],
                    [-0.1540, -0.5100,  0.2317],
                    [-0.5175, -0.0368,  0.2007]], requires_grad=True)
        '''

    def forward(self, input:Tensor):
        x_gate = self.gate(input)
        x_proj = self.proj(input)
        output = x_gate * x_proj + (1-x_gate) * input
        # print(output.size())
        return output


### END YOUR CODE 
if __name__ == '__main__':
    # Seed the Random Number Generators
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    highway = Highway(emb_size=3)
    x = torch.randn(1,1,3)
    y = highway(x)
    print(y)
    assert y.shape == x.shape
    #tensor([[[-0.0743, -0.3612, -0.0601]]], grad_fn=<AddBackward0>)


