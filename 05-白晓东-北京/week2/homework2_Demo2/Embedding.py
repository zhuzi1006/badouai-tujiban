# coding:utf8
import torch
import torch.nn as nn

'''
embedding层的处理
'''

num_embeddings = 26  # 通常对于nlp任务，此参数为字符集字符总数
embedding_dim = 3   # 每个字符向量化后的向量维度
embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
print(embedding_layer.weight, "随机初始化权重")

# 构造输入
x = torch.LongTensor([[1, 2, 3], [2, 2, 0]])
embedding_out = embedding_layer(x)
print(embedding_out)

