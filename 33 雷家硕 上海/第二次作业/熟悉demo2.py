import torch
import torch.nn as nn
import numpy as np
import random
import json

class torchmodel(nn.Module):
    def __init__(self, input_dim, char_len, vocab):
        #凡是各种规定的维度，因为都相同，所以都可以用input_dim表示，
        #线性层需要规定隐藏层的维度
        super(torchmodel, self).__init__()
        self.embedding = nn.Embedding(len(vocab)+1, input_dim)
        #编号从1开始，0行永远不会被抽到，因此还要再补一行，否则选取的时候会超出界限
        self.layer = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(0.2)
        self.activation = torch.sigmoid
        self.pooling = nn.MaxPool1d(char_len)#沿着行数，即字符长度pooling
        self.classify = nn.Linear(input_dim, 3)#几分类output_dim就是几
        self.loss = nn.functional.cross_entropy#多分类任务loss用交叉熵

    def forward(self, x, y = None):
        x = self.embedding(x)#生成的x比如是6*20
        x = self.layer(x)#6*20
        x = self.dropout(x)#6*20
        x = self.activation(x)#6*20
        x = self.pooling(x.transpose(1,2)).squeeze()#1*20
        x = self.classify(x)
        y_pre = self.activation(x)
        if y is not None:
            return self.loss(y_pre, y.squeeze())
        else:
            return y_pre

def built_vocab():
    string = 'abcdefghijklmnopqrstuvwxyz'
    vocab = {}
    for index, i in enumerate(string):
    #这样可以同时获得每个字符的编号和本身
        vocab[i] = index + 1
    vocab['unk'] = len(vocab) + 1
    return vocab

def built_sample(vocab, char_len):
    #sample既包含x，也包含标签y
    #所以既要考虑如何选char_len个字符作为输入，也要设定规则告诉样本是哪类样本
    #首先随机选取char_len个字符作为输入
    x = [random.choice(list(vocab.keys())) for _ in range(char_len)]
    #for后加_表示的是单纯地循环
    #之后规定是正样本还是负样本
    #A类样本：包含abc但不包含xyz
    if set('abc') & set(x) and not set('xyz') & set(x):
        y = 0
    #B类样本：包含xyz但不包含abc
    elif set('xyz') & set(x) and not set('abc') & set(x):
        y = 1
    #C类样本，都不包含
    else:
        y = 2
    #将字符转为序号，方便后面embedding
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y
    #注意这里，每个x都是一个list，而y是一个数

def build_dataset(sample_num, vocab, char_len):
    #注意这里只是构造数据集，到底是用来训练还是预测都可以
    trainset_x = []
    trainset_y = []
    #让x，y分成2大部分，而不是混在一起
    for i in range(sample_num):
        x, y = built_sample(vocab, char_len)
        trainset_x.append(x)
        trainset_y.append([y])
    return torch.LongTensor(trainset_x), torch.LongTensor(trainset_y)
    #都是list，不用FloatTensor

def build_model(char_dim, char_len, vocab):
    model = torchmodel(char_dim, char_len, vocab)
    return model

def evaluation(model, vocab, char_len):
    model.eval()
    sample_num = 200
    x, y = build_dataset(sample_num, vocab, char_len)
    correct, wrong = 0, 0
    y = y.squeeze()#降了一维，是一个一维数组了，但y还是一个tensor
    # print('共有A类样本：%d，B类：%d，C类：%d'%
    #       (np.count_nonzero(i for i in y if y == 0),
    #        np.count_nonzero(i for i in y if y == 1),
    #        np.count_nonzero(i for i in y if y == 2)))
    print('共有A类样本：%d，B类：%d，C类：%d' %
          (y.tolist().count(0), y.tolist().count(1), y.tolist().count(2)))
    #tolist的作用是将内层和外层全部转化为列表，但是tensor也能计数
    #建立评价指标
    with torch.no_grad():
        # 注意这是定义评价方法，即评价已经成型的模型，而不是训练
        # 所以传入的是没有标签的样本，最终结果也就直接是预测值，看与真实值的差距，而不会选择计算loss
        y_pre = model(x)
        for y_p, y_t in zip(y_pre, y):
            if int(torch.argmax(y_pre)) == int(y_t):
            #torch.argmax的意思是取y_pre概率最高的那一维的维数（列数）
                correct += 1
            else:
                wrong += 1
    print('正确预测个数：%d/%d，正确率：%f'%(correct, sample_num, correct/sample_num))
    return correct/sample_num
'''
到现在为止，体系全部建好
可以直接对某个模型进行评价，训不训练只是一个选择
这里还是写一下训练
'''
def main():
    #训练的数量规定
    epoch_num = 15#训练轮数
    sample_size = 1000#每轮训练中样本总数
    batch_size = 20#每轮训练中每次训练的样本个数

    #规定模型需要的各种变量
    char_len = 6
    char_dim = 20

    #建立基础的东西
    vocab = built_vocab()#建立字表
    model = build_model(char_dim, char_len, vocab)#建立模型
    optimizer = torch.optim.Adam(model.parameters())#建立优化器
    '''
    这里没给定学习率，待会可以对比一下
    '''
    #开始运行模型
    for epoch in range(epoch_num):
        '''
        要告诉它这是训练
        '''
        model.train()
        #对于每轮训练
        for batch in range(int(sample_size/batch_size)):
            #对于每次训练，一共50次训练
            #建立每次训练的测试数据集
            #每次训练的数据集都是不一样的
            x, y = build_dataset(batch_size, vocab, char_len)
            loss = model(x, y)
            '''
            加不加forward的对比
            '''
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        evaluation(model, vocab, char_len)
    #不管训练结果如何，训练结束，要先保存训练好的模型（即权重），用于预测
    torch.save(model.state_dict(), 'model.pth')
    #保存字表
    writer = open('vocab.json', 'w', encoding='utf8')
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

def test(inputstrings, vocab_path, model_path):
    # 引入字表
    vocab = json.load(open(vocab_path, 'r', encoding='utf8'))

    #引入模型
    model = build_model(20, 6, vocab)
    #如何更改模型中的矩阵
    model.load_state_dict(torch.load(model_path))

    #引入输入数据
    input = []
    for char in inputstrings:
        x = [vocab[word] for word in char]
        input.append(x)
    input = torch.LongTensor(input)

    #开始预测，但是不需要dropout层，那么只要告诉是测试模式就自动不用dropout了
    model.eval()
    with torch.no_grad():
        y_pre = model.forward(input)
        #注意这里的y_pre包含所有字符串的结果

    #组织结果的形式
    for index, char in enumerate(inputstrings):
        print(int(torch.argmax(y_pre[index])), char, y_pre[index])


if __name__ == '__main__':
    #固定句式
    main()
    # inputstrings = ['jhuacc', 'kwhiud', 'gydshs']
    # pre = test(inputstrings, 'vocab.json', 'model.pth')










