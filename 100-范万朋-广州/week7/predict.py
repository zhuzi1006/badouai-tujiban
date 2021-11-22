# -*- codeing = utf-8 -*-
# @Time: 2021/11/17 23:41
# @Author: 棒棒朋
# @File: predict.py
# @Software: PyCharm
"""
加载模型，对所有准确率大于70%的模型进行测试
"""
import os
import numpy as np
from transformers import BertTokenizer
from model import TorchModel
from config import Config
import torch


def load_vocab(vocab_path):
    """
    加载词表为一个字典
    """
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


class Encode():
    def __init__(self, text, config):
        self.text = text
        self.config = config
        self.index_to_label = {0: '家居', 1: '房产', 2: '股票', 3: '社会', 4: '文化',
                               5: '国际', 6: '教育', 7: '军事', 8: '彩票', 9: '旅游',
                               10: '体育', 11: '科技', 12: '汽车', 13: '健康',
                               14: '娱乐', 15: '财经', 16: '时尚', 17: '游戏'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] in ["bert_cnn", "bert", "bert_lstm"]:
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])  # 词表
        self.config["vocab_size"] = len(self.vocab)  # 词表长度
        # self.load()  # 加载训练数据

    def load(self):
        """
        加载中文句子，转换为数字张量
        """
        if self.config["model_type"] in ["bert_cnn", "bert", "bert_lstm"]:
            # 如果使用的是bert，那就需要使用bert的词表对样本进行编码
            input_id = self.tokenizer.encode(self.text, max_length=self.config["max_length"],
                                             pad_to_max_length=True)
        else:
            input_id = self.encode_sentence(self.text)  # 对每个字都进行编码
        input_id = torch.LongTensor(input_id)  # 将编码后的样本转换为张量
        return input_id

    def encode_sentence(self, text):
        """
        普通的网络的编码，将中文转换为数字
        """
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        """
        补齐或截断输入的序列，使其可以在一个batch内运算
        """
        input_id = input_id[:self.config["max_length"]]  # 只使用句子的前20个字符
        input_id += [0] * (self.config["max_length"] - len(input_id))  # 不足20个字符的句子用0填充
        return input_id


def load_model_path():
    """ 加载模型文件夹中的所有模型文件名称 （放入一个列表中）"""
    model_path_list = []
    for root, dirs, files in os.walk(
            r"E:\Python项目\人工智能\AI_Demo2\2021-11-14\pipeline\output"):
        for file in files:
            model_path_list.append(file)
    return model_path_list


def output(model, model_name, encode, text):
    """
    输出预测结果
    :param model: 模型
    :param model_name: 模型文件的全程
    :param encode: 将文本编码的类对象
    :param text: 输入的测试文本
    """
    # 将字符串进行编码
    text_sentence = encode.load()
    # # 把[20]升为[1,20]
    x = np.full([1, 20], text_sentence)
    result = model.forward(torch.tensor(x))  # 开始预测
    result = torch.argmax(result)  # 选出概率最大的索引值
    print("当前模型：", model_name)
    print("当前测试句子：", text)
    print("预测结果：", encode.index_to_label[result.item()])
    print("************************************************************")


def predict(text):
    """ 预测结果，只使用准确率大于 70%的模型 """
    Config["vocab_size"] = 4623
    Config["class_num"] = 18
    model_list = load_model_path()  # 将所有模型名称都放在列表里面，方便预测
    for i in model_list:  # 循环每一个模型
        p_index = i.find("%.pth")
        try:  # 准确率低于70%
            if float(i[p_index - 5:p_index]) < 70.00:
                continue
        except ValueError:  # 准确率低于10%
            continue
        model_name = i[0:i.find("__lr")]
        Config["model_type"] = model_name
        encode = Encode(text, Config)
        # bert模型
        if model_name in ["bert_cnn", "bert", "bert_lstm"]:
            Config["hidden_size"] = 768
            model = TorchModel(Config)  # 初始化模型
            # 加载训练好的模型权重
            model.load_state_dict(torch.load(
                r"E:\Python项目\人工智能\AI_Demo2\2021-11-14\pipeline\output" + "\\" + i))
            # 输出预测结果
            output(model, i, encode, text)
        else:
            # 其他模型，需要设置隐藏层数量
            for hs in [128, 256, 512]:
                Config["hidden_size"] = hs
                try:
                    model = TorchModel(Config)  # 加载模型
                    model.load_state_dict(torch.load(
                        r"E:\Python项目\人工智能\AI_Demo2\2021-11-14\pipeline\output" + "\\" + i))  # 加载训练好的权重
                except RuntimeError:
                    continue
                output(model, i, encode, text)


if __name__ == '__main__':
    # text = input("请输入要测试的句子:")
    text = "EDG拿下了英雄联盟S11总决赛冠军！！"
    predict(text)
    pass
