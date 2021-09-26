# DNN练习。第一层为5*3，第二层为3*6
import torch
import torch.nn as nn
import numpy as np

# 方法1：torch法
class Torchmodel(nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2):
        super(Torchmodel,self).__init__()
        self.layer1 = nn.Linear(input_size,hidden_size1,bias = False)   # 第一层为5*3
        self.layer2 = nn.Linear(hidden_size1,hidden_size2,bias = False)  # 第二层为3*6

    def forward(self,x):
        x2 = self.layer1(x)
        y = self.layer2(x2)
        return y

# 方法2：使用numpy手动实现
# 公式：y = w*x + b
class Diymodel:
    def __init__(self,weight1,weight2):
        self.weight1 = weight1
        self.weight2 = weight2

    def forward(self,x):
        x2 = np.matmul(x,self.weight1.T)  # 注意，w1存储时，列在前，所以要转置。同理,w2
        y = np.matmul(x2,self.weight2.T)
        return y

x = np.array([1,2,3,5,1])   # 输入
# torch实验
hiddensize1 = 3
hiddensize2 = 6
torch_model = Torchmodel(len(x),hiddensize1,hiddensize2)
x2 = torch.FloatTensor([x])   # torch输入必须为tensor类型
y_out = torch_model.forward(x2)
print(y_out,"torch预测结果")
w1 = torch_model.state_dict()["layer1.weight"].numpy()  # 取出第一层的W，供Diy使用
w2 = torch_model.state_dict()["layer2.weight"].numpy()  # 取出第二层的W，供Diy使用

# Diy实现
diymodel = Diymodel(w1,w2)
y_out2 = diymodel.forward(x)
print(y_out2,"diy预测结果")

