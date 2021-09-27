# embedding层练习
import torch
import torch.nn as nn

# 方法1：直接使用nn.Embedding
num = 9  #此参数为字符集字符总数
embedding_row = 3   #每个字符向量化后的向量行数
layer = nn.Embedding(num, embedding_row) # 构造embedding层
print(layer.weight, "随机初始化后的权重")

#构造输入
x = torch.LongTensor([[1,2,3],[2,4,6],[3,2,5]])  # torch的输入类型必须是tensor
embedding_out = layer(x) # 将转化后的x代入层中
print(embedding_out)


# 方法2：定义类
class Torchmodel(nn.Module):
    def __init__(self,num,embedding_row):
        super(Torchmodel,self).__init__()   # 调用nn.Module中的用法
        self.num = num
        self.embedding_row = embedding_row
        self.layer = nn.Embedding(num, embedding_row)

    def weight(self):
        return self.layer

    def forward(self,x):
        layer = self.layer(x)
        return layer

num = 9  #此参数为字符集字符总数
embedding_row = 3   #每个字符向量化后的向量行数
x = torch.LongTensor([[1,2,3],[2,4,6],[3,2,5]])
model = Torchmodel(num,embedding_row)
print(model.weight().weight,"随机初始化权重")  # 随机初始化权重
y = model.forward(x)
print(y,"输出")