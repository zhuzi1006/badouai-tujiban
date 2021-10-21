import torch
import torch.nn as nn
num_embeddings = 6#字符集的大小
embedding_dim = 3#每个字符向量化后的维度
embedding_layer = nn.Embedding(num_embeddings, embedding_dim)#指定生成的随机初始化的矩阵的行列
print(embedding_layer.weight,"随机初始化矩阵")

x = torch.LongTensor([1,2,5])#给定字符编号,也就是输入
embedding_out = embedding_layer(x)
print(embedding_out)
