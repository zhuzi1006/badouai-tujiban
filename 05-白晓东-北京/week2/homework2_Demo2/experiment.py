"""
from torch import nn
import torch
linear = nn.Linear(in_features=64 * 3, out_features=5)
a = torch.rand(3, 7, 64 * 3)
print(a.shape)
print(linear.weight.shape)
b = linear(a)
print(b.shape)
"""


# import torch as t
# from torch import nn
# # in_features由输入张量的形状决定，out_features则决定了输出张量的形状
# connected_layer = nn.Linear(in_features=64*64*3, out_features=5*5)
# # 假定输入的图像形状为[64,64,3]
# input = t.randn(2, 64, 64, 3)
# # 将四维张量转换为二维张量之后，才能作为全连接层的输入
# input = input.view(2, 64*64*3)
# print(input.shape)
# output = connected_layer(input)  # 调用全连接层
# print(output.shape)


# import numpy as np
# x = np.array([1, 2, 3, 4])
# w = np.array([[1, 2, 3, 4],
#              [2, 3, 4, 5],
#              [3, 4, 5, 3],
#              [4, 5, 2, 1]])
# y = np.dot(x, w)  # np.dot()中一维数组x，可以根据需要为列向量或行向量
# print(y)

# import numpy as np
#
# w = np.array([[-0.1146,  0.2886,  0.1898, -0.4177],
#         [-0.1390, -0.0666,  0.3318,  0.3582],
#         [ 0.2631,  0.2008,  0.1409, -0.1976],
#         [ 0.3388, -0.2081, -0.0101, -0.3440]])
# x = np.array([1, 2, 3, 4])
# y = np.dot(x, w.T)
# print(y)


# 在python中一个汉字算一个字符，一个英文字母算一个字符
# 用 ord() 函数判断单个字符的unicode编码是否大于255即可。
# def is_contain_chinese(check_str):
#     # check_str是一个unicode编码的字符。例如
#     # check_str=u'fff好几个'
#     for ch in check_str:
#         # if u'\u4e00' <= ch <= u'\u9fff':
#         if ord(ch) > 255:
#             print(ch)
# is_contain_chinese(u"sjflfjlwo我爱学习")


# for ch in "wojownf":
#     if ord(ch) > 255:
#         print("找到了")
#         break
# else:
#     print('没找到')


chars = "aabcdefghijklmnopqrstuvwxyz我爱编程编程使我快乐在badou提升自己0123456789"  # 字符集
vocab = {}
index = 1
for char in chars:
    if vocab.get(char, 0) == 0:
        vocab[char] = index  # 每个字对应一个序号
        index += 1
vocab['unk'] = len(vocab) + 1
