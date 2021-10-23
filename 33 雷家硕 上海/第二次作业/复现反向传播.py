'''
为什么torch的结果和diy的结果对不上
下载functional包。。。。。
'''


import torch
import torch.nn as nn
import numpy as np
import copy

class torchmodel(nn.Module):
    def __init__(self, hidden_size):
        super(torchmodel, self).__init__()
        self.layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss

    def forward(self, x, y = None):
        y_pre = self.layer(x)
        y_pre = self.activation(y_pre)
        if y is not None:
            return self.loss(y_pre, y)
        else:
            return y_pre

class diymodel():
    def __init__(self, weight):
        self.weight = weight

    def forward(self, x, y = None):
        y_pre = np.dot(self.weight, x)
        y_pre = self.sigmoid(y_pre)
        if y is not None:
            return self.loss(y_pre, y)
        else:
            return y_pre

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def loss(self, y_pre, y_true):
        return np.sum(np.square(y_pre - y_true))/len(y_pre)

    def diygrad(self, x, y_pre, y):
        grad_loss_sigmoid = 2/len(x) * (y_pre - y)
        grad_sigmoid_wx = y_pre * (1 - y_pre)
        grad_wx_w = x
        grad = grad_loss_sigmoid * grad_sigmoid_wx
        grad = np.dot(grad.reshape(len(x), 1), grad_wx_w.reshape(1, len(x)))
        return grad

def optim(weight, grad):
    alpha = 1e-3  # 学习率
    beta1 = 0.9  # 超参数
    beta2 = 0.999  # 超参数
    eps = 1e-8  # 超参数
    t = 0  # 初始化
    mt = 0  # 初始化
    vt = 0  # 初始化
    # 开始计算
    t = t + 1
    gt = grad
    mt = beta1 * mt + (1 - beta1) * gt
    vt = beta2 * vt + (1 - beta2) * gt ** 2
    mth = mt / (1 - beta1 ** t)
    vth = vt / (1 - beta2 ** t)
    weight = weight - (alpha / (np.sqrt(vth) + eps)) * mth
    return weight

x = np.array([1,2,3,4])
y = np.array([3,2,4,5])

torch_model = torchmodel(len(x))
tensor_x = torch.FloatTensor([x])
tensor_y = torch.FloatTensor([y])
torch_loss = torch_model.forward(tensor_x, tensor_y)
torch_w = torch_model.state_dict()['layer.weight']
print(torch_w, 'torch初始化权重')
numpy_w = copy.deepcopy(torch_w.numpy())

optimizer = torch.optim.Adam(torch_model.parameters())
optimizer.zero_grad()

torch_loss.backward()


optimizer.step()

update_w = torch_model.state_dict()['layer.weight']




diy_model = diymodel(numpy_w)
diy_loss = diy_model.forward(x, y)
print(torch_loss, 'torch_loss')
print(diy_loss, 'diy_loss')
diy_grad = diy_model.diygrad(x, diy_model.forward(x), y)
print(torch_model.layer.weight.grad, 'torch_grad')
print(diy_grad, 'diy_grad')
diy_update_w = optim(numpy_w, diy_grad)
print(update_w, 'torch更新后权重')
print(diy_update_w, 'diy更新后权重')