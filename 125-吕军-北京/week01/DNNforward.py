#coding:utf8

import torch
import torch.nn as nn
import numpy as np
# import keras
# from keras import Sequential

"""
手动实现简单的神经网络
用两种框架实现单层的全连接网络
不使用偏置bias
"""

# class KerasModel:
#     def __init__(self, input_size, hidden_size1, hidden_size2):
#         # input_size
#         # hidden_size1
#         # hidden_size2
#         #顺序模型是多个网络层的线性堆叠
#         self.model = Sequential()
#         self.model.add(keras.layers.Dense(hidden_size1, use_bias=False))
#         self.model.add(keras.layers.Dense(hidden_size2, use_bias=False))
#         self.model.compile(optimizer="adam")
#         self.model.build((1, input_size))
#
#     def forward(self, x):
#         return self.model.predict(x)

class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        # input_size 3 维度
        # hidden_size1 5 维度
        # hidden_size2 2 维度
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1, bias=False)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2, bias=False)

    def forward(self, x):
        hidden = self.layer1(x)
        # print("torch hidden", hidden)
        y_pred = self.layer2(hidden)
        return y_pred

#自定义模型
class DiyModel:
    def __init__(self, weight1, weight2):
        # weight1 5 * 3
        # weight2 2 * 5
        self.weight1 = weight1
        self.weight2 = weight2

    def forward(self, x):
        # x => 1 * 3
        # hidden 1 * 5
        hidden = np.dot(x, self.weight1.T)
        # y_pred 1 * 2
        y_pred = np.dot(hidden, self.weight2.T)
        return y_pred


x = np.array([1, 0, 0])  #网络输入

#torch实验
torch_model = TorchModel(len(x), 5, 2)
print(torch_model.state_dict())
torch_model_w1 = torch_model.state_dict()["layer1.weight"].numpy()
torch_model_w2 = torch_model.state_dict()["layer2.weight"].numpy()
print(torch_model_w1, "torch w1 权重") # 5 * 3
print(torch_model_w2, "torch w2 权重") # 2 * 5
torch_x = torch.FloatTensor([x])
y_pred = torch_model.forward(torch_x)
print("torch模型预测结果：", y_pred) # 1 * 2

diy_model = DiyModel(torch_model_w1, torch_model_w2)
y_pred_diy = diy_model.forward(np.array([x]))
print("diy模型预测结果：", y_pred_diy)
print("-----------------------------")
#keras实验
# keras_model = KerasModel(len(x), 5, 3)
# keras_model_w1 = keras_model.model.get_weights()[0]
# keras_model_w2 = keras_model.model.get_weights()[1]
# print(keras_model_w1, "keras w1 权重")
# print(keras_model_w2, "keras w2 权重")
# y_keras_pred = keras_model.forward(np.array([x]))
# print("keras模型预测结果:", y_keras_pred)
#
# diy_model = DiyModel(keras_model_w1.T, keras_model_w2.T)
# y_pred_diy = diy_model.forward(x)
# print("diy模型预测结果：", y_pred_diy)

