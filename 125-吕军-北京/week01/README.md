numpy
1、axis=0 => 按列相加
   axis=1 => 按行相加
2、二维矩阵的transpose => 转置，行列互换
  三维矩阵的transpose：X轴用0表示，Y轴用1表示；Z轴用2来表示
  transpose（1，0，2）表示X轴与Y轴发生变换

embedding
    nn.Embedding => 初始化embedding矩阵
        num_embeddings 矩阵行数
        embedding_dim  矩阵列数

DNN: 全连接层，线性层 => y=w2.T * (w1.T * x)
RNN: 循环神经网络 => y = tanh(w_ih * xt + w_hh * ht) ht= y
CNN: 卷积神经网络 => y = sum(kernel_weight * window)

  
    