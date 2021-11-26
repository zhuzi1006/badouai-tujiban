import numpy as np
import torch
import torch.nn as nn

class torchmodel(nn.Module):
    def __init__(self, input_size, hidden_size):#这里只定义输入和隐单元，是一个宏观的层面。只有h一个隐单元
        super(torchmodel, self).__init__()
        self.layer = nn.RNN(input_size, hidden_size, bias= False, batch_first= True)
        #定义函数时，参数（权重）不用管，会自己生成，只需要指定权重的维度就可以了。
        #这个层面就比较细了，公式中除了参数还有偏置和隐单元初始值需要设定

    def forward(self, x):
        return self.layer(x)#注意这里没有直接返回预测值

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5]])
torch_model = torchmodel(len(x), 5)
inx = torch.FloatTensor([x])
output, h = torch_model.forward(inx)
print(output.detach(),'rnn预测结果')
print(h.detach(), 'rnn隐含层结果')
torch_weight_ih = torch_model.state_dict()['layer.weight_ih_l0']
troch_weight_hh = torch_model.state_dict()['layer.weight_hh_l0']

#diy
class diyrnn():
    def __init__(self, weighthh, weightih, hidden_size):
        self.weighthh = weighthh
        self.weightih = weightih
        self.hidden_size = hidden_size

    def forward(self, x):
        listhidden = []
        hiddenin = np.zeros(self.hidden_size)
        for xt in x:#逐行读取x
            # wh = np.dot(self.weighthh, hiddenin)
            # ux = np.dot(self.weightih, xt)
            wh = np.dot(hidden, self.weighthh.T)
            ux = np.dot(xt, self.weightih.T)
            #计算就用这个带转置的公式，妈的
            hidden = np.tanh(wh + ux)
            listhidden.append(hidden)
            hiddenin = hidden
        return np.array(listhidden), listhidden[-1]
        #要np.array一下，否则出来的是list
        #最终的隐含层是一个整体的矩阵，只不过他是一行行迭代得到的
        #最终的预测输出结果就是这个隐含层矩阵的最后一行
        #所以本质上是求一个隐藏层矩阵作为最终的输出。所以说一开始定义torch函数的时候除输入只需要定义一个hidden_size


diy_model = diyrnn(troch_weight_hh, torch_weight_ih, 5)
h_diy, y_diypre = diy_model.forward(x)
print(h_diy,'diy模型隐含层结果')
print(y_diypre, 'diy模型预测结果')




