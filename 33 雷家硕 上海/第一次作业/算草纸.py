import numpy as np
import torch
import torch.nn as nn


# print(x[0])
# a = np.array([x[0][0:2]])
# print(a)
# x = np.array([[0.1, 0.2, 0.3, 0.4],
#               [-3, -4, -5, -6],
#               [5.1, 6.2, 7.3, 8.4],
#               [-0.7, -0.8, -0.9, -1]])
# inx = torch.FloatTensor([[x]])
# print(inx[0][0][0][0], type(inx[0][0][0][0]))
import random
inputlen = 3
char = 'abcdhjeu'
dic = {}
for index, char in enumerate(char):
    dic[char] = index + 1
dic['unk'] = len(dic) + 1
# print(dic)
x = [random.choice(list(dic.keys())) for i in range(inputlen)]
# print(x, type(x))
inx = [dic.get(i, dic['unk']) for i in x]
print(inx)
# input_h = x.shape[0]#4
# input_d = x.shape[1]#4
# kernel_size = 2
# kernel_weight = np.array([[-0.1, -0.2 ],
#                           [-0.3, -0.4]])
#
# y = np.zeros([input_h+1-kernel_size, input_d+1-kernel_size])
# for j in range(input_h+1-kernel_size):
#     for i in range(input_d+1-kernel_size):
#         list_result_row = []
#         for k in kernel_weight:#这不是1*几的矩阵，就是一行数
#             x_row = x[j][i:i+kernel_size]
#             result_row = x_row * k
#             list_result_row.append(result_row)
#         print(list_result_row)
#         y[j][i] = np.sum(list_result_row[:])
# print(y)
# print(len(x))
# y = np.array([[1,2],
#               [2,3],
#               [3,4]])
# z = np.array([[2],
#               [1]])
# f = np.dot(y, z)
# print(f)
# q = np.array([[1,2,1],
#               [2,3,1]])
# c = np.array([[2,3,2],
#               [3,4,1]])
# mul = q*c#对位相乘的方法
# sum = np.sum(mul[:])
# print(q[0][2])#调取第1行第3列
# y = np.zeros([2,3])
# y[0][0] = 1
# print(y)
