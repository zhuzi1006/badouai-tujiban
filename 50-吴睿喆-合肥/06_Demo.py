#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现

举例： 指定字为 “我”
我来自中国 - 正样本  
他来自中国 - 负样本 

知道规则的情况下： 只要“我”在字符串中就是正样本 
在不知道规则的情况下：给出一批正样本，同时也给出一批负样本。
训练后，来辨别正样本和负样本。
------------------------------------------------
以下代码使用模型判断 abc 三个字符串是否在字符串中。
监督训练模型，网络功能的变化依赖于训练数据生成函数 build_sample 
中正样本和负样本的规则。
------------------------------------------------
目标：
（1）将代码进行修改，或者将结构进行修改 （比较懒，目前不想修改）
（2）将判断的条件进行修改 （条件就在 build_sample 函数中）
（3）注意模型每层中输入输出的变化情况 （代码中的注释已经很清楚了）

    x = self.embedding(x)  #input shape:(batch_size, sen_len)               
    x = self.layer(x)      #input shape:(batch_size, sen_len, input_dim)   
    上面的部分明显表明  embedding 层的输入为  (batch_size, sen_len), 
    输出为 (batch_size, sen_len, input_dim)
        
"""

class TorchModel(nn.Module):
    def __init__(self, input_dim, sentence_length, vocab):
        # 继承一个基础类 nn.Module ，并在此类中搭建一些网络层
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab) + 1, input_dim)      #(27+1,20) , Embedding 共具有28种类型输出，每个输出为 20 长度的向量
        # 网络层中一定需要使用矩阵运算，因此首先需要对字符做 Embedding
        self.layer = nn.Linear(input_dim, input_dim)                  #(20,20)
        # 线性层，也可以称为全链接层，计算公式为 y = W * x + b
        # 其中 W 和 b 是参与训练的参数 , W 的维度决定了隐含层输出的维度，
        # 一般称为隐单元个数 hidden size
        self.pool = nn.AvgPool1d(sentence_length)                     # 一维平均池化层，每六个层化一次
        # 平均池化层，用于缩减模型的大小，提高计算速度
        self.classify = nn.Linear(input_dim, 1)                       # (20,1)
        # 用于分辨的分类层，输出维度为 1， 输入维度为 input_dim
        self.activation = torch.sigmoid     #sigmoid做激活函数
        # 激活函数层，选用 sigmoid 函数
        self.dropout = nn.Dropout(0.1)
        # dropout 层，用于随机输出结果
        self.loss = nn.functional.mse_loss  #loss采用均方差损失
        # 损失函数使用 mse 均方根误差来衡量


    #定义此层的前向计算过程，可以通过前向计算过程获取到实际的数据情况
    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  #input shape:(batch_size, sen_len)
        x = self.layer(x)      #input shape:(batch_size, sen_len, input_dim)
        x = self.dropout(x)    #input shape:(batch_size, sen_len, input_dim)
        x = self.activation(x) #input shape:(batch_size, sen_len, input_dim)
        x = self.pool(x.transpose(1,2)).squeeze() #input shape:(batch_size, sen_len, input_dim)
        x = self.classify(x)   #input shape:(batch_size, input_dim)
        y_pred = self.activation(x)         #input shape:(batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值可以同真实值计算得到 Loss
        else:
            return y_pred


#字符集随便挑了一些汉字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    '''
    创建符号集合，从 26 个字母从 1 - 26 创建索引 ，
    对于字符串  unk ，则创建索引为 28
    :return:  创建完的字典 vocab
    '''
    chars = "abcdefghijklmnopqrstuvwxyz"  #字符集 ，可以使用 a-z 形成字符串
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index + 1   #每个字对应一个序号
    vocab['unk'] = len(vocab)+1
    return vocab


#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):
    #随机从字表选取 sentence_length 个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 将 vocab 的键值转换为列表的形式，通过生成表达式生成 sentence_length 个字

    #指定哪些字出现时为正样本
    if set("abc") & set(x):   # 只要包含 abc 中的任何一个就是正样本
        y = 1
    #指定字都未出现，则为负样本
    else:
        y = 0
    #使用set函数将“abc"转换为集合{'a', 'c', 'b'}，只要 x 包含一个就可以成为正样本

    # 将字转换成序号，为了做embedding
    x = [vocab.get(word, vocab['unk']) for word in x]
    # 由字典的 get 方法获取到样本 x 中每个字符的数值标号，最后以 array 的形式返回
    return x, y


#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    '''
    :param sample_length:   需要生成样本的数量
    :param vocab:           生成的字符集
    :param sentence_length: 生成样本的字符长度
    :return:                返回生成的数据集
    '''
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])

    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length):
    # 使用之前定义的 TorchModel 类，创建类实例
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    '''
    如果模型中有BN层(Batch Normalization）和Dropout，在测试时添加model.eval()。
    model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变。
    对于Dropout，model.eval()是利用到了所有网络连接，即不进行随机舍弃神经元。
    '''
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), 200 - sum(y)))
    correct, wrong = 0, 0

    # Note: 借助经典的 with-open 操作来理解此处的 with torch.no_grad() 操作
    with torch.no_grad():      #不使用梯度下降算法，仅仅用于测试
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1   #负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1   #正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    # - - - - - - - - 训练前的超参数配置 - - - - - - - - -
    epoch_num = 10        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 1000   #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    # - - - - - - - - - - - - - - - - - - - - - - - - -
    vocab = build_vocab()       #建立字表
    model = build_model(vocab, char_dim, sentence_length)    #建立模型
    optim = torch.optim.Adam(model.parameters(), lr=0.005)   #建立优化器
    log = []
    # 生成模型数据，并且训练模型
    for epoch in range(epoch_num):
        # model.train()的作用是启用 Batch Normalization 和 Dropout。
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length) #构建一组训练样本
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    plt.plot(range(len(log)), [l[0] for l in log])  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log])  #画loss曲线
    plt.show()
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#最终预测，借助已有 Torch 模型结果进行预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)    #建立模型
    model.load_state_dict(torch.load(model_path))       #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式，不使用dropout
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        print(round(float(result[i])), input_string, result[i]) #打印结果

if __name__ == "__main__":
    # main()
    test_strings = ["abvxee", "casdfg", "rqweqg", "nlkdww"]
    predict("model.pth", "vocab.json", test_strings)
