#coding:utf8
import torch
import torch.nn as nn

'''
embedding层的处理
'''

num_embeddings = 6  #通常对于nlp任务，此参数为字符集字符总数
embedding_dim = 3   #每个字符向量化后的向量维度
embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
print(embedding_layer.weight, "随机初始化权重")


#            embedding_layer.weight
# [                           embedding_dim          ]
# [                                 |                ]
# [                                \/                ]
# [                   [[-1.3684,  0.5654, -0.3678],  ]
# [                    [ 0.7762, -0.2808, -0.2345],  ]
# [ num_embeddings ->  [-1.1074,  0.2539,  0.7048],  ]
# [                    [ 0.6644,  0.1788,  1.2770],  ]
# [                    [-0.8151, -0.9899, -0.2315],  ]
# [                    [ 1.4344,  0.1786,  1.2606]]  ]

#构造输入
x = torch.LongTensor([[1,2,3],[2,2,0]])
embedding_out = embedding_layer(x)
print(embedding_out)

#                             embedding_layer
# [                                                                          ]
# [                                            [[-1.3684,  0.5654, -0.3678], ]
# [                       num_embeddings        [ 0.7762, -0.2808, -0.2345], ]                  embedding_dim
# [            one_hot    [[0 1 0 0 0 0]        [-1.1074,  0.2539,  0.7048], ]          [[ 0.7762, -0.2808, -0.2345],
# [ [1 2 3]  ---------->   [0 0 1 0 0 0],   *   [ 0.6644,  0.1788,  1.2770], ]   =       [-1.1074,  0.2539,  0.7048],
# [                        [0 0 0 1 0 0]],      [-0.8151, -0.9899, -0.2315], ]           [ 0.6644,  0.1788,  1.2770]]
# [                                             [ 1.4344,  0.1786,  1.2606]] ]
# [                                                                          ]
# [                                                                          ]
