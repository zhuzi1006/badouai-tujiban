#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import copy

"""
基于Pytorch的网络编写
手动实现梯度计算和反向传播
加入激活函数
"""


class TorchModel(nn.Module):

    def __init__(self, hidden_size):
        super(TorchModel, self).__init__()
        # https://zhuanlan.zhihu.com/p/152198144(线性层的入门理解)
        # https://blog.csdn.net/qq_42079689/article/details/102873766 (进一步理解，主要看评论，输入也可以是三维，看experiment.py)
        self.layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss  # loss采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.layer(x)
        y_pred = self.activation(y_pred)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


# 自定义模型，接受一个参数矩阵作为入参
class DiyModel:
    def __init__(self, weight):
        self.weight = weight

    def forward(self, x, y=None):
        y_pred = np.dot(self.weight, x)
        y_pred = self.diy_sigmoid(y_pred)
        if y is not None:
            return self.diy_mse_loss(y_pred, y)
        else:
            return y_pred

    # sigmoid
    def diy_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 手动实现mse，均方差loss
    def diy_mse_loss(self, y_pred, y_true):
        return np.sum(np.square(y_pred - y_true)) / len(y_pred)

    # 手动实现梯度计算
    def calculate_grad(self, y_pred, y_true, x):

        # 反向传播更新梯度
        # 均方差损失函数为 (y_pred - y_true) ^ 2 / n ，其导数为   2 * (y_pred - y_true) / n
        grad_loss_sigmoid_wx = 2/len(x) * (y_pred - y_true)
        # sigmoid函数 y = 1/(1+e^(-x)) 的导数 = y * (1 - y)
        grad_sigmoid_wx_wx = y_pred * (1 - y_pred)
        # wx对w求导 = x
        grad_wx_w = x
        # 导数链式相乘
        grad = grad_loss_sigmoid_wx * grad_sigmoid_wx_wx
        grad = np.dot(grad.reshape(len(x),1), grad_wx_w.reshape(1,len(x)))
        return grad


# 梯度更新
def diy_sgd(grad, weight, learning_rate):
    return weight - grad * learning_rate


# adam梯度更新
def diy_adam(grad, weight):
    # 参数应当放在外面，此处为保持后方代码整洁简单实现一步
    alpha = 1e-3   # 学习率
    beta1 = 0.9    # 超参数
    beta2 = 0.999  # 超参数
    eps = 1e-8     # 超参数
    t = 0          # 初始化
    mt = 0         # 初始化
    vt = 0         # 初始化
    # 开始计算
    t = t + 1
    gt = grad
    mt = beta1 * mt + (1 - beta1) * gt  # 将之前的梯度也考虑了进来
    vt = beta2 * vt + (1 - beta2) * gt ** 2
    mth = mt / (1 - beta1 ** t)  # 越来越相信mth本身
    vth = vt / (1 - beta2 ** t)
    weight = weight - (alpha / (np.sqrt(vth) + eps)) * mth  # 学习率的自动变化
    return weight


# 输入和输出
x = np.array([1, 2, 3, 4])  # 输入
y = np.array([3, 2, 4, 5])  # 预期输出

# 得到由一个线性层、激活层组成的网络结构的初始化权重
torch_model = TorchModel(len(x))
torch_model_w = torch_model.state_dict()["layer.weight"]
print(torch_model_w, "初始化权重")
numpy_model_w = copy.deepcopy(torch_model_w.numpy())

# 将输入和预期的输出转化为tensor
torch_x = torch.FloatTensor([x])
torch_y = torch.FloatTensor([y])

# torch的前向计算过程，得到loss
torch_loss = torch_model.forward(torch_x, torch_y)
print("torch模型计算loss：", torch_loss)

# 自定义模型的前向计算过程，前向计算一次得到loss， 手动实现loss计算
diy_model = DiyModel(numpy_model_w)
diy_loss = diy_model.forward(x, y)
print("diy模型计算loss：", diy_loss)

# torch中优化器的实现方式
learning_rate = 0.1
print(torch_model.parameters())
# optimizer_SGD = torch.optim.SGD(torch_model.parameters(), lr=learning_rate)  # 将torch_model中的所有参数传进来
optimizer_Adam = torch.optim.Adam(torch_model.parameters())
optimizer_Adam.zero_grad()  # 将优化器的梯度先置为0

# pytorch的反向传播操作
torch_loss.backward()
print(torch_model.layer.weight.grad, "torch 计算梯度")  # 查看某层权重的梯度

# 手动实现反向传播
grad = diy_model.calculate_grad(diy_model.forward(x), y, x)
print(grad, "diy 计算梯度")

# torch梯度更新
optimizer_Adam.step()
# 查看更新后权重
update_torch_model_w = torch_model.state_dict()["layer.weight"]
print(update_torch_model_w, "torch更新后权重")

# 手动梯度更新
# diy_update_w = diy_sgd(grad, numpy_model_w, learning_rate)
diy_update_w = diy_adam(grad, numpy_model_w)
print(diy_update_w, "diy更新权重")