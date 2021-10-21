# 熟悉embedding
# import torch
# import torch.nn as nn
# import numpy as np
#
# num_embedding = 10
# embedding_dim = 4
# embedding_layer = nn.Embedding(num_embedding, embedding_dim)
# #这个layer就是一个model，它的输入是x，并不是num_embedding和embedding_dim
# print(embedding_layer.weight, '随机初始化权重矩阵')
#
# x = torch.LongTensor([1,2,7])
# embedding_out = embedding_layer(x)
# print(embedding_out)



# 熟悉DNN
# import numpy as np
# import torch
# import torch.nn as nn
#
# #torch
# class torchmodel(nn.Module):
#     def __init__(self, input_size, hidden1_size, hidden2_size):
#         super(torchmodel, self).__init__()
#         self.layer1 = nn.Linear(input_size, hidden1_size, bias= False)
#         self.layer2 = nn.Linear(hidden1_size, hidden2_size, bias= False)
#
#     def forward(self, x):
#         hidden = self.layer1(x)
#         y_pre = self.layer2(hidden)
#         return y_pre
# #diy
# class diymodel():
#     def __init__(self, weight1, weight2):
#         self.weight1 = weight1
#         self.weight2 = weight2
#
#     def forward(self, x):
#         hidden = np.dot(x, self.weight1.T)
#         y_diypre = np.dot(hidden, self.weight2.T)
#         return y_diypre
#
# #开始运行
# x = np.array([1,2,3])
#
# #torchrun
# torch_model = torchmodel(len(x), 6, 4)
# inx = torch.FloatTensor([x])
# y_torch = torch_model.forward(inx)
# print(y_torch, 'torch训练结果')
# torch_weight1 = torch_model.state_dict()['layer1.weight']
# torch_weight2 = torch_model.state_dict()['layer2.weight']
# #diyrun
# diy_model = diymodel(torch_weight1, torch_weight2)
# y_diy = diy_model.forward(x)
# print(y_diy, 'diy训练结果')

#熟悉RNN
# import torch
# import torch.nn as nn
# import numpy as np
# class torchrnn(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(torchrnn, self).__init__()
#         self.layer = nn.RNN(input_size, hidden_size, bias= False, batch_first= True)
#
#     def forward(self, x):
#         return self.layer(x)
# class diyrnn():
#     def __init__(self, weighthh, weightih, hidden_size):
#         self.weighthh = weighthh
#         self.weightih = weightih
#         self.hidden_size = hidden_size
#
#     def forward(self, x):
#         listhidden = []
#         hidden = np.zeros(self.hidden_size)
#         for xt in x:
#             # wh = np.dot(self.weighthh, hidden)
#             # ux = np.dot(self.weightih, xt)
#             wh = np.dot(hidden, self.weighthh.T)
#             ux = np.dot(xt, self.weightih.T)
#             hidden_next = np.tanh(wh + ux)
#             listhidden.append(hidden_next)
#             hidden = hidden_next
#         return np.array(listhidden), listhidden[-1]
#
# # x = np.array([[1,2,3],
# #               [2,3,4],
# #               [3,4,5],
# #               [4,5,6]])
#
# x = np.array([[1,2,3,4],
#              [2,3,4,5]])
# #torchrun
# input_size = 4#注意len和维度不一样，这里的维度就一直是列数
# hidden_size = 3
# torch_model = torchrnn(input_size, hidden_size)
# inx = torch.FloatTensor([x])
# torch_out, torch_h = torch_model.forward(inx)
# y_pre = torch_out.detach().numpy()
# h = torch_h.detach().numpy()
# print(y_pre, 'torch result')
# print(h, 'torch hidden matrix')
# torch_weight_hh = torch_model.state_dict()['layer.weight_hh_l0']
# torch_weight_ih = torch_model.state_dict()['layer.weight_ih_l0']
# print(torch_weight_hh, '隐单元权重矩阵')
# print(torch_weight_ih, '输入权重矩阵')
# #diyrun
# diy_model = diyrnn(torch_weight_hh, torch_weight_ih, hidden_size)
# y_diypre, diyh = diy_model.forward(x)
# print(y_diypre, 'diy result')
# print(diyh, 'diy hidden matirx')

#熟悉CNN
# import torch
# import torch.nn as nn
# import numpy as np
# class torchcnn(nn.Module):
#     def __init__(self, input_channel_num, kernel_num, kernel_size):
#         super(torchcnn, self).__init__()
#         self.layer = nn.Conv2d(input_channel_num, kernel_num, kernel_size, bias= False)
#
#     def forward(self, x):
#         return self.layer(x)
#
# class diycnn():
#     def __init__(self, input_h, input_d, kernel_size, kernel_weight):
#         self.input_h = input_h
#         self.input_d = input_d
#         self.kernel_size = kernel_size
#         self.kernel_weight = kernel_weight
#
#     def forward(self, x):
#         output = []
#         for w in self.kernel_weight:
#             y = np.zeros((self.input_h + 1 - self.kernel_size, self.input_d + 1 - self.kernel_size))
#             weight = w.squeeze().numpy()
#             for i in range(self.input_h+1-self.kernel_size):
#                 for j in range(self.input_d+1-self.kernel_size):
#                     window = x[i:i+self.kernel_size, j:j+self.kernel_size]
#                     y[i][j] = np.sum(window * weight)
#             output.append(y)
#         return np.array(output)
#
# x = np.array([[0.1, 0.2, 0.3, 0.4],
#               [-3, -4, -5, -6],
#               [5.1, 6.2, 7.3, 8.4],
#               [-0.7, -0.8, -0.9, -1]])
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

# #diy
# diy_model = diycnn(x.shape[0], x.shape[1], kernel_size, torch_weight)
# diyout = diy_model.forward(x)
# print(diyout, 'diy输出')

#熟悉demo
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
