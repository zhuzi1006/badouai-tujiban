# RNN练习
import torch
import torch.nn as nn
import numpy as np

# torch 实现
class Torchmodel(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(Torchmodel,self).__init__()
        self.layer = nn.RNN(input_size,hidden_size,bias = False,batch_first=True)

    def forward(self,x):
        return self.layer(x)
# numpy实现
class Diymodel:
    def __init__(self,W,U,hidden_size):
        self.W = W
        self.U = U
        self.hidden_size = hidden_size
    def forward(self,x):
        ht = np.zeros((self.hidden_size))
        output = []
        for xt in x:
            Ux = np.matmul(self.U,xt)
            Wx = np.matmul(self.W,ht)
            h_next = np.tanh(Ux + Wx)
            output.append(h_next)
            ht = h_next
        return np.array(output),ht


x = np.array([[1,2,3,4,5],
              [2,4,6,8,0],
              [3,2,5,7,8],
              [5,3,1,7,9],
              [6,1,3,5,9],
              [5,4,2,8,9]])
hiddensize = 5  # 代表h的个数
model1 = Torchmodel(5,hiddensize) # torchmodel第一个参数为x的列数
torch_x = torch.FloatTensor([x])
y_out,h = model1.forward(torch_x)
print(y_out.detach().numpy(),"RNN预测结果")
print(h.detach().numpy(),"RNN预测隐含层结果")

# diy实验
U = model1.state_dict()["layer.weight_ih_l0"]
W = model1.state_dict()["layer.weight_hh_l0"]
model2 = Diymodel(W,U,hiddensize)
y_out2,h2 = model2.forward(x)
print(y_out2,"diy预测结果")
print(h2,"diy预测隐含层结果")