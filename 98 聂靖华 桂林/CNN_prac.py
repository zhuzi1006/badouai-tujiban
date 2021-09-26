# CNN练习
import torch
import torch.nn as nn
import numpy as np

# 方法1：torch实现
class Torchmodel(nn.Module):
    def __init__(self, in_channel, out_channel, kernel):
        super(Torchmodel, self).__init__()
        self.layer = nn.Conv2d(in_channel, out_channel, kernel, bias=False)

    def forward(self, x):
        return self.layer(x)

#方法2：numpy实现
class DiyModel:
    def __init__(self, height, width, weight, kernel_size):
        self.height = height
        self.width = width
        self.weights = weight
        self.kernel_size = kernel_size

    def forward(self, x):
        output = []
        for W_i in self.weights:
            W_i = W_i.squeeze().numpy() # W_i要参与后面的计算，所以转化为矩阵
            output2 = np.zeros((self.height-kernel_size+1, self.width-kernel_size+1)) # 每个卷积核的输出
            for i in range(self.height-kernel_size+1):
                for j in range(self.width-kernel_size+1):
                    temp = x[i:i+kernel_size, j:j+kernel_size]   # 卷积核覆盖的子矩阵
                    output2[i, j] = np.sum(W_i * temp) # 此处kernel_weight * window为矩阵的点乘。
            output.append(output2)
        return np.array(output)


x = np.array([[1,2,3,4,5],
              [2,4,6,8,0],
              [3,2,5,7,8],
              [5,3,1,7,9],
              [6,1,3,5,9],
              [5,4,2,8,9]])  #网络输入

#torch实验
in_channel = 1   # 输入个数
out_channel = 3  # 输出个数，等于卷积核数量
kernel_size = 3  # 卷积核的大小
model1 = Torchmodel(in_channel, out_channel, kernel_size)
w = model1.state_dict()["layer.weight"]    # 取出每个卷积核的权重
torch_x = torch.FloatTensor([[x]])    # tensor类型才能输入torch模型
y_out = model1.forward(torch_x)
print(y_out, "torch模型预测结果")
print(w,"卷积核的权重")

# diy，numpy实现
model2 = DiyModel(6, 5, w, kernel_size)
y_out2 = model2.forward(x)
print(y_out2, "diy模型预测结果")

