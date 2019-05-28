#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch.nn as nn
import torch.nn.functional as F
### YOUR CODE HERE for part 1i
from torch import Tensor
import torch

class CNN(nn.Module):
    def __init__(self,word_emb_size,char_emb_size,max_word_length,window_size=3):
        super(CNN,self).__init__()
        self.conv = nn.Conv1d(in_channels=char_emb_size,out_channels=word_emb_size,kernel_size=window_size)
        self.max_pool = nn.MaxPool1d(max_word_length-window_size+1)


    def forward(self, input:Tensor):
        #input (b,seq,max_word_length,char_emb)
        in_shape = input.shape
        X = input.view(in_shape[0]*in_shape[1],in_shape[2],-1)
        X_trans = X.permute(0,2,1)
        X_conv = self.conv(X_trans)
        X_convout = self.max_pool(F.relu(X_conv))
        # print(X_convout.size())
        input.squeeze()
        X_cnn = X_convout.squeeze(2).view(in_shape[0],in_shape[1],-1)
        # print(X_cnn.size())
        return X_cnn
if __name__ == '__main__':
    word_emb_size = 256
    char_emb_size = 50
    max_word_length = 21
    window_size = 5
    cnn = CNN(word_emb_size,char_emb_size,max_word_length,window_size)
    x = torch.randn(32,10,21,50)
    y = cnn(x)
    print(y.size())




### END YOUR CODE

