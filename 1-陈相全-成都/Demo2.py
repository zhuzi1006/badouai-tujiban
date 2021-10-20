#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
# import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现

"""

class TorchModel(nn.Module):
    #模型的基本参数：embedding层，线性层（输入，隐藏，输出），池化层
    def __init__(self, input_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()#继承nn.module里面的类
        self.embedding = nn.Embedding(len(vocab) + 1, input_dim)
        #创建嵌入矩阵,索引个数（种类）为前者，每个索引维度是后者
        #len(vocab) + 1是这个embedding函数，需要预留一位来作为未知字符的保留（此处的未知字符部不是指unk，unk是我们自己定义的未知字符）
        self.layer = nn.Linear(input_dim, input_dim+5)
        self.layer2 = nn.Linear(input_dim+ 5, input_dim+5)
        self.pool = nn.MaxPool1d(sentence_length)
        self.classify = nn.Linear(input_dim+5, 4)
        self.activation = torch.nn.ReLU()     #relu做激活函数
        self.dropout = nn.Dropout(0.5)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  #input shape:(batch_size, sen_len) (10,6)
        x = self.layer(x)      #input shape:(batch_size, sen_len, input_dim) (10,6,20)
        x = self.dropout(x)    #input shape:(batch_size, sen_len, input_dim)
        x = self.activation(x) #input shape:(batch_size, sen_len, input_dim)

        x = self.layer2(x)
        x = self.dropout(x)
        x = self.activation(x)

        x = self.pool(x.transpose(1,2)).squeeze() #input shape:(batch_size, sen_len, input_dim)
        # transpose转置，squeeze（a）把a所有维度为1的维度删除
        x = self.classify(x)   #input shape:(batch_size, input_dim)  (10,6)
        y_pred = self.activation(x)               #input shape:(batch_size, 1) (10,1)
        if y is not None:#y非0，则计算loss值，并把值返回main函数
            return self.loss(y_pred, y.squeeze())
        else:
            return y_pred

#字符集随便挑了一些汉字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index + 1   #每个字对应一个序号
    print("--------------------------")
    print(vocab)
    print("--------------------------")

    vocab['unk'] = len(vocab)+1
    print("--------------------------")
    print(vocab)
    print("--------------------------")

    print("只取键:",end="")
    print(vocab.keys())
    print("列表化后：", end="")
    print(list(vocab.keys()))
    return vocab

#随机生成一个样本
#这就是在专门生成单个样本进行判断了
#从所有字中选取sentence_length个字
#都没有的话为负样本
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    #random.choice() 方法返回一个列表，元组或字符串的随机项。
    #A类样本
    if set("abc") & set(x) and not set("xyz") & set(x) and not set("wtf") & set(x):#仅有abc
        #&是与运算，与abc比较
        #set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
        y = 0
    #B类样本
    elif not set("abc") & set(x) and set("xyz") & set(x) and not set("wtf") & set(x):#仅有xyz
        y = 1
    # C类样本
    elif not set("abc") & set(x) and not set("xyz") & set(x) and set("wtf") & set(x):#仅有wtf
        y = 2
    #D类样本
    else:
        y = 3

    # 将字转换成序号，为了做embedding
    x = [vocab.get(word, vocab['unk']) for word in x]
    #dict.get(key, default=None)
    # get() 函数返回指定键的值，如果键不在字典中返回默认值 None 或者设置的默认值。

    #print(x)打印出来就是形如
    # [10, 24, 20, 9, 26, 15]
    # [10, 2, 13, 2, 26, 5]
    # [27, 2, 22, 24, 26, 16]
    # [3, 2, 19, 9, 12, 19]
    # [27, 23, 24, 23, 26, 15]
    #这种东西，就是先化为序号


    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    #                  20        已建立          6
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    #print(dataset_y)
    #形如
    # [[0], [2], [2], [1], [1], [0], [2], [0], [2], [0], [0], [0], [2], [2], [1], [1], [1], [1], [2], [2]]
    # [[2], [0], [1], [0], [0], [1], [1], [2], [2], [1], [2], [0], [2], [1], [0], [2], [2], [0], [2], [2]]
    #print(dataset_x)
    # 形如
    #[[11, 27, 7, 13, 17, 21], [20, 24, 27, 7, 24, 9], [15, 2, 1, 25, 5, 20],。。。。。。

    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)
    #torch.Tensor默认是torch.FloatTensor是32位浮点类型数据，torch.LongTensor是64位整型

#建立模型
#为什么不直接用TorchModel(char_dim, sentence_length, vocab)
#反而麻烦一下将其包装一下呢
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


#测试集
#测试代码
#用来测试每轮模型的准确率
def evaluate(model,           vocab,   sample_length):
    #       训练之后的model      表           6
    model.eval()
    total = 500 #测试样本数量
    x, y = build_dataset(total, vocab, sample_length)   #建立200个用于测试的样本，也是随机搞的
    y = y.squeeze()#去掉为1的维度
    print("以下为测试集情况：")
    print("A类样本数量：%d, B类样本数量：%d, C类样本数量：%d, D类样本数量：%d"%(y.tolist().count(0), y.tolist().count(1), y.tolist().count(2),y.tolist().count(3)))
    #tolist()转化成列表，然后count（“x”）数x这个东西有多少个
    correct, wrong = 0, 0

    '''
    requires_grad: 如果需要为张量计算梯度，则为True，否则为False。我们使用pytorch创建tensor时，可以指定requires_grad为True（默认为False），
    grad_fn： grad_fn用来记录变量是怎么来的，方便计算梯度，y = x*3,grad_fn记录了y由x计算的过程。
    grad：当执行完了backward()之后，通过x.grad查看x的梯度值。
    '''
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            # print(list(zip(y_pred, y)))
            # [(tensor([4.0332e-05, 9.9951e-01, 9.9814e-01]), tensor(1)), (tensor([0.0994, 0.0904, 0.9364]),
            #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
            #在 Python 3.x 中为了减少内存，zip() 返回的是一个对象。如需展示列表，需手动 list() 转换
            #markdown文件里面有zip笔记

            if int(torch.argmax(y_p)) == int(y_t):
                #torch.argmax()返回指定维度最大值的序号
                correct += 1   #正样本判断正确
            else:
                wrong += 1

    print("正确预测个数：%d / %d, 正确率：%f"%(correct, total, correct/(correct+wrong)))
    return correct/(correct+wrong)


#初始化，进行训练操作
def main():
    epoch_num = 15        #训练轮数
    batch_size = 20       #每轮训练有多组，这是每组训练的样本个数
    train_sample = 3000   #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度（特征）
    sentence_length = 6   #样本文本长度
    vocab = build_vocab()       #建立字表
    model = build_model(vocab, char_dim, sentence_length)    #建立模型
    optim = torch.optim.Adam(model.parameters(), lr=0.005)   #建立优化器
    log = []
    for epoch in range(epoch_num):
        model.train()#库里面的训练函数
        '''
        在使用pytorch构建神经网络的时候，训练过程中会在程序上方添加一句
        model.train():作用是启用batch normalization和drop out。
        model.eval():测试过程中会使用model.eval()，这时神经网络会沿用batch normalization的值，并不使用drop out。
        '''
        watch_loss = []#记录所有轮数中计算出来的loss值
        for batch in range(int(train_sample / batch_size)):#算出来是分50组
            x, y = build_dataset(batch_size, vocab, sentence_length) #构建一组训练样本
            optim.zero_grad()    #清空过往梯度
            loss = model(x, y)   #计算loss：model->build_model->TorchModel
            loss.backward()      #计算梯度：这好像是内置的函数,反向传播，计算当前梯度
            optim.step()         #根据梯度更新网络参数
            #print(type(loss))
            #>>>  <class 'torch.Tensor'>多维矩阵
            watch_loss.append(loss.item())#item用来将tensor格式转化为python的数据类型格式。把数据从tensor取出来，变成python的数据类型，方便后续处理
            #print(type(loss.item()))
            #>>>  <class 'float'>
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        #                                    epoch从零开始    求平均
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果，evaluate返回正确率
        log.append([acc, np.mean(watch_loss)])#log日志：存储每一大轮的准确率和每一大轮的几十组的几十个loss的平均值
        # print(log)
        # [[0.49, 1.0799714016914368], [0.45, 1.0530911242961885], [0.45, 1.040774906873703],。。。。。。。
    # plt.plot(range(len(log)), [l[0] for l in log])  #画acc曲线
    # plt.plot(range(len(log)), [l[1] for l in log])  #画loss曲线
    # plt.show()

    #保存模型
    torch.save(model.state_dict(), "model.pth")
    #pth是保存模型的后缀名，还有ckpt，pkl等格式可以保存
    '''https://blog.csdn.net/qq_27009517/article/details/111272115?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163438357716780264063280%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=163438357716780264063280&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-111272115.pc_search_result_control_group&utm_term=pkl%E5%92%8Cckpt&spm=1018.2226.3001.4187'''
    #torch.nn.Module模块中的state_dict变量存放训练过程中需要学习的权重和偏执系数

    # 保存词表进json文件
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    #这是因为json.dumps 序列化时默认使用ascii编码.想输出真正的字符需要指定ensure_ascii=False：
    #indent缩进空格
    writer.close()
    #写完了关掉文件
    return



#最终预测：训练完了，现在通过之前训练好的模型，加载进来，然后输入数据，让模型自己判断
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度,理解为特征，即20个特征表示一个字
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
        print(int(torch.argmax(result[i])), input_string, result[i]) #打印结果


#使用
if __name__ == "__main__":
    main()#训练，测试
    test_strings = ["juvxee", "yrwfrg", "rbweqg", "nlhdww"]#样本
    predict("model.pth", "vocab.json", test_strings)#判断