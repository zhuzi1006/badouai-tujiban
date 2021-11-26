import numpy as np
import torch
import torch.nn as nn

class TorchCNN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel):
        #以图片为例
        # in_channel是输入的图片的通道数，灰度图就是单通道，为1。RGB是3通道，为3
        #out_channel是卷积也就是卷积核的个数
        #kernel是卷积核的大小
        super(TorchCNN, self).__init__()
        self.layer = nn.Conv2d(in_channel, out_channel, kernel, bias=False)#不管哪个神经网络模型都有偏置

    def forward(self, x):
        return self.layer(x)

#自定义CNN模型
class DiyModel:
    def __init__(self, input_height, input_width, weights, kernel_size):
        self.height = input_height
        self.width = input_width
        self.weights = weights
        self.kernel_size = kernel_size

    def forward(self, x):
        output = []
        for kernel_weight in self.weights:
            kernel_weight = kernel_weight.squeeze().numpy() #shape : 2x2
            #squeeze的作用就是降维，降一维
            kernel_output = np.zeros((self.height - kernel_size + 1, self.width - kernel_size + 1))
            for i in range(self.height - kernel_size + 1):
                for j in range(self.width - kernel_size + 1):
                    window = x[i:i+kernel_size, j:j+kernel_size]#丛矩阵中取一个矩阵原来可以这样直接取
                    kernel_output[i, j] = np.sum(kernel_weight * window)
                    # np.dot(a, b) != a * b。矩阵也可以这样直接对位相乘
            output.append(kernel_output)
        return np.array(output)


x = np.array([[0.1, 0.2, 0.3, 0.4],
              [-3, -4, -5, -6],
              [5.1, 6.2, 7.3, 8.4],
              [-0.7, -0.8, -0.9, -1]])  #网络输入

#torch实验
in_channel = 1
out_channel = 3#3个核这里输出的也是3个y，没做后续的处理
kernel_size = 3
torch_model = TorchCNN(in_channel, out_channel, kernel_size)
# print(torch_model.state_dict())
torch_w = torch_model.state_dict()["layer.weight"]#所有的权重矩阵都是用state_dict()调
print(torch_w, torch_w.shape, '原始的权重矩阵')#因为它不是矩阵，而是一个包含了：卷积核数 ，输入通道数，二维矩阵的一个4维张量
# print(torch_w.numpy(), type(torch_w.numpy()), 'numpy后的权重张量')#numpy后并不会改变维度，只是从tensor转变成了ndarry
torch_x = torch.FloatTensor([[x]])
output = torch_model.forward(torch_x)
output = output.detach().numpy()
print(output, "torch模型预测结果\n")
print("---------------")
diy_model = DiyModel(x.shape[0], x.shape[1], torch_w, kernel_size)
output = diy_model.forward(x)
print(output, "diy模型预测结果")
# input_channel = 3#单通道
# output_channel = 3#有3个卷积核
# kernel_size = 3#卷积核为3*3
#
# #torch多通道
# torch_x = torch.FloatTensor([[x]])
# torchoutlist = []
# torchweightlist = []
# for i in range(input_channel):
#     torch_model = torchcnn(1, output_channel, kernel_size)
#     torchout = torch_model.forward(torch_x)
#     torchout = torchout.detach()
#     torchoutlist.append(torchout)
#     torch_weight = torch_model.state_dict()['layer.weight']
#     torchweightlist.append(torch_weight)
# print('torch输出:', torchoutlist,  '\n', 'torch核的权重:', torchweightlist)