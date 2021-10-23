#coding:utf8

import torch
import torch.nn as nn
import numpy as np


"""
基于pytorch的网络编写
测试dropout层
"""

import torch

x = torch.Tensor([1,2,3,4,5,6,7,8,9])
dp_layer = torch.nn.Dropout(0.5)
dp_x = dp_layer(x)
print(dp_x)

#                       droput 50%
# [1,2,3,4,5,6,7,8,9]   -------->   [1 0 0 4 0 0 7 0 9]   *   1 / (1 - 0.5)    =     [ 2.  0.  0.  8.  0.  0. 14.  0.  18.]

