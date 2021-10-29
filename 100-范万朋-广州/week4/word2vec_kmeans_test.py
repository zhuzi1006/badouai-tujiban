# -*- codeing = utf-8 -*-
# @Time: 2021/10/29 15:40
# @Author: 棒棒朋
# @File: word2vec_kmeans_test.py
# @Software: PyCharm

"""
# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
使用 欧氏距离 / 余弦相似度 计算类内平均距离
只输出类内平均距离前 10（从小到大排列）

"""
import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


def load_word2vec_model(path):
    """
    # 输入模型文件路径
    # 加载训练好的模型
    :param path:
    :return:
    """
    model = Word2Vec.load(path)
    return model


def load_sentence(path):
    """
    加载所有标题，并且进行切词
    :param path:
    :return:
    """
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences


def sentences_to_vectors(sentences, model):
    """
    将文本向量化, 先把句子中的每个字的向量都加起来，加起来之后求平均值作为句子的向量
    :param model:
    :param sentences: 切词好后的句子set集合
    :return: 向量化之后的句子集合
    """
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)

        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)

        vectors.append(vector / len(words))
    return np.array(vectors)


def distance(vector1, vector2, dist_name='EUCLIDEAN'):
    """
    计算向量的距离, 每个向量100维
    :param dist_name: 使用哪种距离
    :param vector1:
    :param vector2:
    :return: 两个向量的距离
    """
    if 'EUCLIDEAN' == dist_name:  # 使用欧氏距离
        tmp = 0
        for i in range(len(vector1)):
            tmp += pow(vector1[i] - vector2[i], 2)

        return pow(tmp, 0.5)
    elif 'COSINE' == dist_name:  # 使用余弦相似度距离
        x_dot_y = sum([x * y for x, y in zip(vector1, vector2)])
        sqrt_x = math.sqrt(sum([x ** 2 for x in vector1]))
        sqrt_y = math.sqrt(sum([x ** 2 for x in vector2]))
        if sqrt_y == 0 or sqrt_y == 0:
            return 0
        return x_dot_y / (sqrt_x * sqrt_y)


def main():
    model = load_word2vec_model("model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题，并且切好词。len = 1796
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化 shape = (1796, 100)

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    # defaultdict() 产生一个带有默认值的dict。主要针对key不存在的情况下，也希望有返回值的情况。
    sentence_label_dict = defaultdict(list)  # 初始化一个句子字典
    vector_label_dict = defaultdict(list)  # 初始化一个句子向量字典
    euclidean_label_dict = defaultdict(list)  # 初始化一个欧氏距离字典

    for center_vector, label in zip(kmeans.cluster_centers_, kmeans.labels_):  # 取出中心向量和标签
        vector_label_dict[label].append(center_vector)  # 把中心向量都放在每个类别字典的第0个

    for vector, label in zip(vectors, kmeans.labels_):  # 取出句子向量和标签
        vector_label_dict[label].append(vector)  # 同标签的句子向量放到一起（index从1开始）

    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起

    for label in vector_label_dict:  # 取出句子向量和标签
        dist = 0  # 每个类别的类内平均距离默认为0
        for j in range(len(vector_label_dict[label])):
            if j == 0:  # 跳过中心向量自己跟自己比较
                continue
            dist += distance(vector_label_dict[label][0], vector_label_dict[label][j], 'COSINE')

        dist = dist / (len(vector_label_dict[label]) - 1)  # 计算该类别的平均距离
        euclidean_label_dict[label].append(dist)  # 将该类别的平均距离添加到字典，用于后面排序

    # 对字典进行排序（按欧氏距离 从小到大排序）
    euclidean_label_dict = sorted(euclidean_label_dict.items(), key=lambda kv: kv[1])

    for i in range(10):
        # 只打印出类内平均距离最小的前10个
        label = euclidean_label_dict[i][0]  # 标签
        dist = euclidean_label_dict[i][1][0]  # 类内平均距离（使用欧氏距离计算）
        print("---------")
        print("类别: %s " % label, "\t类内平均距离：%f" % dist)
        for j in range(min(15, len(sentence_label_dict[label]))):  # 每个类别最多只打印15条
            print(sentence_label_dict[label][j].replace(" ", ""))


if __name__ == "__main__":
    main()
