import numpy as np
import torch
from torch.autograd import Variable

# a = np.array([[1,2,3],
#               [4,5,6]])#2行3列，维度取小的，是行的个数，是2，类型是ndarry
# x = np.array([[1,2],
#              [2,3],
#              [3,4]])#维度是列的个数，是2
# print(a)
# print(a.shape)
# print(np.sum(a))#所有元素全部加在一起
# print(np.sum(a,axis = 0))#axis = 0是按列
# print(np.sum(a, axis = 1))
# print(np.reshape(a, (1,6)))
# print(np.sqrt(a))#每个元素都开方
# print(np.square(a))#每个元素都平方
# print(np.exp(a))
# print(a.transpose())#转秩
# print(a.flatten())
# print(x.ndim)
# print(np.zeros((2,3)))#2*3的矩阵
# print(np.zeros(2,3,4))#2个3*4的矩阵，这是一个2*3*4的张量了
# print(np.random.rand(2,3))#基于均匀分布随机生成2*3的矩阵
# print(np.random.rand(2,3,4))#2个3*4的张量
# print(np.random.randn(2,3))#基于正态分布随机生成的2*3矩阵
q = np.array([[1,2,3],
              [2,3,4]])
q = torch.FloatTensor(q)#调试的时候红点是不运行的
# print(torch.exp(q))
# print(torch.sum(q, dim = 0))#列和。torch只能和dim一组，np只能和axis一组
# print(q.transpose(1,0))
print(q.flatten())




