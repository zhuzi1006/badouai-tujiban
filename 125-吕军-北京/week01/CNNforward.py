#coding:utf8

import torch
import torch.nn as nn
import numpy as np


"""
手动实现简单的神经网络
使用pytorch实现CNN
手动实现CNN
对比
"""

class TorchCNN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel):
        super(TorchCNN, self).__init__()
        # nn.Conv2d是二维卷积方法，相对应的还有一维卷积方法nn.Conv1d,常用于文本数据的处理，而nn.Conv2d一般用于二维图像
        # in_channel 输入图像通道数 1
        # out_channel 卷积产生的通道数 3
        # kernel_size 卷积核尺寸，可以设为1个int型数或者一个(int, int)型的元组 2
        self.layer = nn.Conv2d(in_channel, out_channel, kernel, bias=False)

    def forward(self, x):
        return self.layer(x)

#自定义CNN模型
class DiyModel:
    def __init__(self, input_height, input_width, weights, kernel_size):
        # input_height 行数  4
        # input_width 列数  4
        # weights 权重.shape (3, 1, 2, 2)
        # kernel_size 2
        self.height = input_height
        self.width = input_width
        self.weights = weights
        self.kernel_size = kernel_size

    def forward(self, x):
        output = []
        for kernel_weight in self.weights:
            # numpy.squeeze() 这个函数的作用是去掉矩阵里维度为1的维度 由 1x2x2 更新为 2x2
            kernel_weight = kernel_weight.squeeze().numpy() #shape : 2x2
            # kernel_output shape 3x3 的0矩阵
            kernel_output = np.zeros((self.height - kernel_size + 1, self.width - kernel_size + 1))
            for i in range(self.height - kernel_size + 1):
                for j in range(self.width - kernel_size + 1):
                    # 得到2 * 2的矩阵
                    window = x[i:i+kernel_size, j:j+kernel_size]
                    # print("+++++++++++++++++++++++++++1")
                    # print(window)
                    # print(kernel_weight)
                    # print("+++++++++++++++++++++++++++2")
                    # print(kernel_weight * window)
                    # print("+++++++++++++++++++++++++++3")
                    kernel_output[i, j] = np.sum(kernel_weight * window) # np.dot(a, b) != a * b
            output.append(kernel_output)
        return np.array(output)


x = np.array([[0.1, 0.2, 0.3, 0.4],
              [-3, -4, -5, -6],
              [5.1, 6.2, 7.3, 8.4],
              [-0.7, -0.8, -0.9, -1]])  #网络输入

#torch实验
in_channel = 1
out_channel = 3
kernel_size = 2
torch_model = TorchCNN(in_channel, out_channel, kernel_size)
print(torch_model.state_dict())
torch_w = torch_model.state_dict()["layer.weight"]
print(torch_w.numpy().shape) # (3, 1, 2, 2)
torch_x = torch.FloatTensor([[x]]) # (4, 4)
output = torch_model.forward(torch_x)
output = output.detach().numpy()
print(output, output.shape, "torch模型预测结果\n") # (1, 3, 3, 3)
print("---------------")
diy_model = DiyModel(x.shape[0], x.shape[1], torch_w, kernel_size)
output = diy_model.forward(x)
print(output, output.shape,  "diy模型预测结果") # (3, 3, 3)

