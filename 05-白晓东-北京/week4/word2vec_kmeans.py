#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences


# 将单句向量化
def single_sentence_to_vectors(sentence, model):
    vectors = []
    words = sentence.split()  #sentence是分好词的，空格分开
    vector = np.zeros(model.vector_size)
    #所有词的向量相加求平均，作为句子向量
    for word in words:
        try:
            vector += model.wv[word]
        except KeyError:
            #部分词在训练中未出现，用全0向量代替
            vector += np.zeros(model.vector_size)
    vectors.append(vector / len(words))
    return np.array(vectors)


#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("/Users/wenjiazhai/Documents/GitHub/badouai-tujiban/07-翟文嘉-南京/week_04/titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    # 计算类内平均距离，并打印结果
    distance_dict = {}
    for label, sentences in sentence_label_dict.items():
        average_vector = np.zeros(model.vector_size)
        distance = 0
        for sentence in sentences:
            single_sentence_vector = single_sentence_to_vectors(sentence, model)
            average_vector += single_sentence_vector.squeeze()
        average_vector /= len(sentences)
        for sentence in sentences:
            text_vector = single_sentence_to_vectors(sentence, model)
            distance += np.sqrt(np.sum(np.square(average_vector - text_vector.squeeze())))
        distance /= len(sentences)
        distance_dict[distance] = sentences
    distance_list = sorted(distance_dict.keys(), key=lambda key:key)
    for i in range(10):
        print("类内平均距离 %s :" % distance_list[i])
        contents = distance_dict[distance_list[i]]
        # print("该label下的几个例子，查看相似性：{}" .format(content))
        for content in contents:  #随便打印几个，太多了看不过来
            print(content.replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()