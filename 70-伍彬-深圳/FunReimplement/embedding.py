import torch
import torch.nn as nn

x = torch.IntTensor([[1, 2, 3],[4, 5, 6]])      # x 中的数代表字典里字的index， 所以不能用默认是float值


"""
num_embeddings (int) - 嵌入字典的大小
embedding_dim (int) - 每个嵌入向量的大小
padding_idx (int, optional) - 如果提供的话，输出遇到此下标时用零填充
max_norm (float, optional) - 如果提供的话，会重新归一化词嵌入，使它们的范数小于提供的值
norm_type (float, optional) - 对于max_norm选项计算p范数时的p
scale_grad_by_freq (boolean, optional) - 如果提供的话，会根据字典中单词频率缩放梯度
"""
eb = nn.Embedding(7, 3)                         # 第一个参数代表字典大小，x中的值从词典中寻址，所以必须大于max(x)， 从0开始
print(eb(x))


