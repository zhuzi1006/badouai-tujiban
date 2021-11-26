import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

#第一部分——定义
class TorchModel(nn.Module):
    #定义模型，分两步：定义有哪些层def __init__；定义它们的组成顺序def forward
    def __init__(self, input_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab) + 1, input_dim)
        self.layer = nn.Linear(input_dim, input_dim)
        self.pool = nn.AvgPool1d(sentence_length)
        self.classify = nn.Linear(input_dim, 1)
        self.activation = torch.sigmoid     #sigmoid做激活函数
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.mse_loss  #loss采用均方差损失


    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  #input shape:(batch_size, sen_len)
        # 比如有10个字符串向量，每个字符串有6个字符，那么输入这个embedding层的矩阵就是（10*6）
        x = self.layer(x)      #input shape:(batch_size, sen_len, input_dim)
        #embedding层输出的作为线性层的输入，embedding层的随机初始化的矩阵是一个26*20的大矩阵
        #假如指定embedding层的维度是20，那么embedding层的输出就是（10，6，20）的张量
        #这里线性层的作用是：
        x = self.dropout(x)    #input shape:(batch_size, sen_len, input_dim)
        #把张量中的元素随机变成0，但不会改变张量的形状
        x = self.activation(x) #input shape:(batch_size, sen_len, input_dim)
        #也不会改变张量的形状，只是对一些数值进行变换
        x = self.pool(x.transpose(1,2)).squeeze() #input shape:(batch_size, sen_len, input_dim)
        #有10个样本，一个一个分析
        #每一个样本最后的输出是一个概率，也就是指定字符在不在这里面的概率。
        #首先将20维那一维池化，就得到了10*6的矩阵（10个样本，每个样本是个6*1的矩阵）
        #之后再通过一个线性层classify就能把6维的映射到1维上了，即一个数
        x = self.classify(x)   #input shape:(batch_size, input_dim)
        y_pred = self.activation(x)               #input shape:(batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def build_vocab():
    char = 'abcdefghijklmnopqrstuvwxyz'
    #因为在放入embedding层之前需要找到样本字符对应的数字编号，所以要构造一个一一对应的字典
    vocab = {}
    for index, char in enumerate(char):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab) + 1
    #字符'unk'的作用是：占位，因为取的时候毕竟还是按index取的，为例不超出list range，所以加了一个元素
    return vocab

# def build_vocab():
#     chars = "abcdefghijklmnopqrstuvwxyz"  #字符集
#     vocab = {}
#     for index, char in enumerate(chars):#enumerate函数就是给出字符及其对应的编号
#         vocab[char] = index + 1   #每个字对应一个序号
#     vocab['unk'] = len(vocab)+1
#     return vocab







#第二部分——准备数据
def build_sample(sample_len, vocab):
    #这是一个字符的生成函数，所以不用管数量
    #database里面不能只有x，还要有标签y（1/0）
    x = [random.choice(list(vocab.keys())) for _ in range(sample_len)]
    #这里只选出来了字符，字符可能会重，同时也没给出相应的编号
    #这里的x直接返回来一个list，每一个字符是一个元素
    if set('abc') & set(x):
        y = 1
    else:
        y = 0
    inx = [vocab.get(i, vocab['unk']) for i in x]
    #直接用vocab[]调取键值不行吗？get函数有什么必要性吗？
    #这个unk咋又出现了？
    return inx,y

# def build_sample(vocab, sentence_length):
#     #随机从字表选取sentence_length个字，可能重复
#     x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
#     #指定哪些字出现时为正样本
#     if set("abc") & set(x):
#         y = 1
#     #指定字都未出现，则为负样本
#     else:
#         y = 0
#     x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
#     return x, y

def build_database(sample_num, sample_len, vocab):#括号里的自变量可以直接拿来用了
    #样本个数，样本长度，字表
    database_x = []
    database_y = []
    #x,y要分成2个单独的数据集
    for i in range(sample_num):
        # x = bulid_sample(sample_len, vocab)[0]
        # y = bulid_sample(sample_len, vocab)[1]
        x, y = build_sample(sample_len, vocab)#直接这样就可以
        database_x.append(x)
        database_y.append([y])#x是矩阵，所以y也要是
    return torch.LongTensor(database_x), torch.FloatTensor(database_y)
    #注意返回的要是tensor，不能是numpy，因为后面直接用这个数据集计算了
    #这个tensor的类型不能轻易改变，必须是Long

# def build_dataset(sample_length, vocab, sentence_length):
#     dataset_x = []
#     dataset_y = []
#     for i in range(sample_length):
#         x, y = build_sample(vocab, sentence_length)
#         dataset_x.append(x)
#         dataset_y.append([y])
#     return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)






#第三部分——指定模型
def build_model(vocab, char_dim, input_len):#这一步是要定义一个模型，不是要输出什么
    #字表，每一个字符的维度，每句话的字符个数
    model = TorchModel(char_dim, input_len, vocab)
    return model
    #这一步有什么意义吗？我直接引用TorchModel不可以吗？

# def build_model(vocab, char_dim, sentence_length):
#     model = TorchModel(char_dim, sentence_length, vocab)
#     #model模型是基于TorchModel类包装的，直接看这个类class
#     return model








#第四部分——设立评价指标
def evaluate(model, vocab, sample_length):#给出准确率
#模型，字表，样本长度
    #通知
    model.eval()#告诉程序即将进入测试


    #给数据
    x, y = build_database(200, sample_length, vocab)
          #建立200个用于测试的样本，直接给数了，所以不需要sample_num这个变量了
    print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), 200 - sum(y)))


    #评价
    correct, wrong = 0, 0
    with torch.no_grad():#记下来，不让求导的意思
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #对于每一对预测值与真实值，都要进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                # 这个0.5的值的设定应该是根据筛选的严格程度来的吧？
                correct += 1   #负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1   #正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    # print要在return之前
    return correct/(correct+wrong)








#第五部分——训练
def main():#自己熟悉
    #测试相关参数设置
    #训练轮数
    epoch_num = 20
    #每轮训练的样本总数
    train_num = 2000
    #每轮训练中每次训练的样本数
    batch_num = 50


    #模型相关准备
    char_dim = 20
    char_num = 6
    vocab = build_vocab()#建立字表
    model = build_model(vocab, char_dim, char_num)#准备模型
    optim = torch.optim.Adam(model.parameters(), lr=0.005)#建立优化器


    #开始训练
    epoch_loss = []
    for epoch in range(epoch_num):#对于每一轮
        model.train()#通知训练开始
        batch_loss = []
        for batch in range(int(train_num/batch_num)):#对于每一次训练(样本是不一样的，要不训练个啥)
            x, y = build_database(batch_num, char_num, vocab)#构建了20个样本
            #梯度归0
            optim.zero_grad()
            #计算损失
            loss = model(x, y)#这里的y是样本中的y，不是建模所需的变量
            #计算梯度
            loss.backward()
            #更新权重
            optim.step()
            batch_loss.append(loss.item())
        print('第%d轮训练的损失为：%f' % (epoch + 1, np.mean(batch_loss)))
        #每一轮训练的损失不是取最后一次的损失，而是取所有次数的损失的平均值
        #除loss外，还要应用评价指标对该轮训练做出评价
        acc = evaluate(model, vocab, char_num)
        epoch_loss.append([acc, np.mean(batch_loss)])
    #训练结束

    # 可视化训练效果
    plt.plot(range(len(epoch_loss)), [h[0] for h in epoch_loss])  # 画acc曲线
    plt.plot(range(len(epoch_loss)), [h[1] for h in epoch_loss])  # 画loss曲线
    plt.show()

    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # torch.save(model.state_dict(), 'model.pth')
    #训练的是权重，所以保存权重就可以了，模型只是一个外壳

    # 保存词表
    writer = open("../vocab.json", "w", encoding="utf8")#新建一个可写文件
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))#把字表写进去
    #ensure_ascii表示的是：
    #indent表示的是：

    # writer.close()
    # writer = open('vocab.json', 'w', encoding = 'utf8')
    # writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    # writer.close()
    return









#第六部分——测试
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))#引入字表
    model = build_model(vocab, char_dim, sentence_length)    #引入模型
    model.load_state_dict(torch.load(model_path))       #将模型中的随机权重改为训练好的权重

    # char_dim = 20
    # sentence_length = 6
    # vocab = json.load(open(vocab_path, 'r', encoding='utf8'))
    # model = build_model(vocab, sentence_length, char_dim)
    # model.load_state_dict(torch.load(model_path))

    # x = []
    # for input_string in input_strings:
    #     x.append([vocab[char] for char in input_string])  #将输入序列化
    # model.eval()   #测试模式，不使用dropout
    # with torch.no_grad():  #不计算梯度
    #     result = model.forward(torch.LongTensor(x))  #模型预测
    # print(result)
    # for i, input_string in enumerate(input_strings):
    #     print(round(float(result[i])), input_string, result[i]) #打印结果
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])#输入序列化
    with torch.no_grad():#调用模型
        result = model.forward(torch.LongTensor(x))#这里的result是所有句子的结果,是概率
    for index, input_string in enumerate(input_strings):
        print(round(float(result[index])), input_string, result[index])
        #round是四舍五入保留整数


if __name__ == "__main__":
    main()#这两步是什么意思？
    test_strings = ["sffxee", "casdfg", "rqweqg", "nlkdww"]
    predict("../model.pth", "vocab.json", test_strings)




