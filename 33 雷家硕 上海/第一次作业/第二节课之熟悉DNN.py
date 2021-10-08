import torch
import torch.nn as nn
import numpy as np

import random
import matplotlib

class TorchModel(nn.Module):
    #这个class是定义一个神经网络模型固定的程序init,self; forward
    #这个模型因为是继承了nn中的模块，所以括号里才有东西，就是被继承的东西
    #被继承的torch的模块可以直接使用
    def __init__(self, input_size, hidden_size1, hidden_size2):#定义TorchModel本身可以实现什么功能，只需要考虑输入和隐单元的大小，不用管参数
        super(TorchModel, self).__init__()#继承模型才需要配套super
        self.layer1 = nn.Linear(input_size, hidden_size1, bias=False)#定义这个层该如何运算
        self.layer2 = nn.Linear(hidden_size1, hidden_size2, bias=False)
        # self就是TorchModel,super底下是构造这个模型有哪些层
        # 层名就是self后面的名字，比如这一层叫layer1
        #这些层都是打包好的函数，所以直接输入系数就好了，而不是真正的矩阵

    def forward(self, x):#定义TorchModel中的一个函数，最后的效果就是可以TorchModel.forward(...)
        hidden = self.layer1(x)
        # print("torch hidden", hidden)
        y_pred = self.layer2(hidden)
        return y_pred
        #这是在规定输入在这些层之间的传递过程

x = np.array([1,0,0])

#调用torch结果展示
#模型调用部分
torch_model = TorchModel(len(x), 5, 2)
inx = torch.FloatTensor([x])#为什么要转成tensor
y = torch_model.forward(inx)
#结果展示部分
print(torch_model.state_dict())#这里可以看到pytorch存矩阵是反着存的
torch_weight1 = torch_model.state_dict()['layer1.weight']
torch_weight2 = torch_model.state_dict()['layer2.weight']
print(torch_weight1,'\n',torch_weight2)
print(y,'预测值')



#自定义模型
class DiyModel:
    #没有继承任何模块，所以要自己运算
    def __init__(self, weight1, weight2):#同理，只需要管因单元
        self.weight1 = weight1
        self.weight2 = weight2

    def forward(self, x):
        hidden = np.dot(x, self.weight1.T)#注意pytorch存储是行列相反的，所以要转置一下
        y_pre = np.dot(hidden, self.weight2.T)
        return y_pre

diy_model = DiyModel(torch_weight1, torch_weight2)
y_diy = diy_model.forward(x)
print(y_diy, 'diy模型预测值')




x = np.array([1, 0, 0])  #网络输入

#torch实验
torch_model = TorchModel(len(x), 5, 2)
print(torch_model.state_dict())
torch_model_w1 = torch_model.state_dict()["layer1.weight"].numpy()
    #取该层权重，权重储存在state_dict()这个字典里，“layer1.weight”是它的键，权重是对应的键值
torch_model_w2 = torch_model.state_dict()["layer2.weight"].numpy()
print(torch_model_w1, "torch w1 权重")
print(torch_model_w2, "torch w2 权重")
torch_x = torch.FloatTensor([x])#因为后面要计算梯度，也就是要求导，所以需要是tensor
y_pred = torch_model.forward(torch_x)
print("torch模型预测结果：", y_pred)

#diy模型实验
diy_model = DiyModel(torch_model_w1, torch_model_w2)
y_pred_diy = diy_model.forward(np.array([x]))
print("diy模型预测结果：", y_pred_diy)
print("-----------------------------")