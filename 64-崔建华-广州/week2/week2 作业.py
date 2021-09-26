# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 21:47:44 2021

@author: Tracy
"""

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, input_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab) + 1, input_dim) #tracy:构造词向量，一个词一行，行数：词语的数目，列数：用m个维度去表示一个词
        self.layer = nn.Linear(input_dim, input_dim) #tracy:分别是输入样本的大小/输出样本的大小
        self.pool = nn.AvgPool1d(sentence_length)
        self.classify = nn.Linear(input_dim, 1)
        self.activation = torch.sigmoid     #sigmoid做激活函数
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.mse_loss  #loss采用均方差损失


    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  #input shape:(batch_size, sen_len)
        x = self.layer(x)      #input shape:(batch_size, sen_len, input_dim)
        x = self.dropout(x)    #input shape:(batch_size, sen_len, input_dim)
        x = self.activation(x) #input shape:(batch_size, sen_len, input_dim)
        x = self.pool(x.transpose(1,2)).squeeze() #input shape:(batch_size, sen_len, input_dim)
        x = self.classify(x)   #input shape:(batch_size, input_dim)
        y_pred = self.activation(x)               #input shape:(batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred
        

# 创建字符集,为将文字转换成数字做准备
def build_vocab():
    chars = "用户流失预测是个什么东西"
    # 去除标点符号
    for i in "',?.!。，？！'":
        chars = chars.replace(i,"")
    vocab = {}
    for index, char in enumerate(chars):
        if vocab.get(char):
            continue
        else:
            vocab[char] = index + 1
    vocab['unk'] = len(vocab) + 1
    return vocab

# 随机生成一个样本
def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    
    if set("用户")&set(x):
        y = 1
    else:
        y  = 0
    x = [vocab.get(word,vocab['unk']) for word in x]
    return x, y

# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

# 测试代码
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    print("本次预测集共有%d个正样本，%d个负样本"%(sum(y), 200 - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y): # 与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct +=1 # 负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1 # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d，正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def main():
    epoch_num = 10 #训练轮数
    batch_size = 20 #每次训练样本个数
    train_sample = 1000 #每轮训练总共训练的样本总数
    char_dim = 20 #每个字的维度
    sentence_length = 6 #样本文本长度
    vocab = build_vocab() #建立字表
    model = build_model(vocab, char_dim, sentence_length)
    optim = torch.optim.Adam(model.parameters(),lr=0.005) #建立优化器
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length) #构建一组训练样本
            optim.zero_grad() #梯度归零
            loss = model(x, y) #计算loss
            loss.backward() #计算梯度
            optim.step() #更新权重
            watch_loss.append(loss.item())
        
        print("==========\n滴%d轮平均loss:%f"% (epoch+1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])
    plt.plot(range(len(log)), [l[0] for l in log]) #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log]) #画loss曲线
    plt.show()
    
    #保存模型
    torch.save(model.state_dict(),"model.pth")
    
    #保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    
    return None

#最终预测
def predict(model_path, vacab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    vocab = json.load(open(vacab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path)) #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char,vocab['unk']) for char in input_string])
    model.eval() #测试模型，不适用dropout
    with torch.no_grad(): #不计算梯度
        result = model.forward(torch.LongTensor(x))
    print("最终预测:\n")
    for i, input_string in enumerate(input_strings):
        print(round(float(result[i])), input_string, result[i])
        
if __name__ == "__main__":
    main()
    test_strings = ["会员积分累计", "会员销售总计", "所有衡量指标"]
    predict("model.pth", "vocab.json", test_strings)
        
        
            
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    
