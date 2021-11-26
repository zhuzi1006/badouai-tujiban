# -*- codeing = utf-8 -*-
# @Time: 2021/11/24 14:38
# @Author: 棒棒朋
# @File: faq_system.py
# @Software: PyCharm
"""
    第八周作业提交
    高频问题库的智能客服系统
"""
import logging
import torch
from config import Config
from model import SiameseNetwork, choose_optimizer
from evaluate import Evaluator
from loader import load_data, DataGenerator, load_schema
import numpy as np


def pre_config(config):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    cuda_flag = torch.cuda.is_available()  # 标识是否使用gpu

    # 对输入的句子进行编码
    encode = DataGenerator(config["train_data_path"], config)

    model = SiameseNetwork(config)  # 初始化模型
    # 加载训练好的模型
    model.load_state_dict(torch.load(
        r"E:\Python项目\人工智能\AI_Demo2\2021-11-21\sentence_match_as_sentence_encoder\model_output\epoch=10__acc=0.87.pth"))

    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    evaluator.knwb_to_vector()
    return model, evaluator, encode


def predict(model, evaluator, encode, dict, string):
    str_encode = encode.encode_sentence(string)
    # 把[20]升为[1,20]
    x = torch.tensor(np.full([1, 20], str_encode))
    x = x.cuda()

    with torch.no_grad():
        test_vectors = model.forward(x)  # 不输入labels，使用模型当前参数进行预测
    res = torch.mm(test_vectors.unsqueeze(0), evaluator.knwb_vectors.T)  # 128x[128,1878] = [1,1878]
    hit_index = int(torch.argmax(res.squeeze()))  # 命中问题标号
    hit_index = evaluator.question_index_to_standard_question_index[hit_index]  # 转化成标准问编号
    return dict.get(hit_index)


if __name__ == '__main__':
    index_to_target = {}
    dict_list = load_schema(Config["schema_path"])
    target_dict = dict([val, key] for key, val in dict_list.items())
    model, evaluator, encode = pre_config(Config)
    while True:
        question = input("请输入问题：")
        res = predict(model, evaluator, encode, target_dict, question)
        print("命中问题：", res)
        print("-----------")
