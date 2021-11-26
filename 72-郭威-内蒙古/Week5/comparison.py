#!/usr/bin/env python3  
#coding: utf-8

#使用基于词向量的分类器
#对比几种模型的效果

import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict


LABELS = {'健康': 0, '军事': 1, '房产': 2, '社会': 3, '国际': 4, '旅游': 5, '彩票': 6, '时尚': 7, '文化': 8, '汽车': 9, '体育': 10, '家居': 11, '教育': 12, '娱乐': 13, '科技': 14, '股票': 15, '游戏': 16, '财经': 17}

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

#加载数据集
def load_sentence(path, model):
    sentences = []
    labels = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = json.loads(line)
            title, content = line["title"], line["content"]
            sentences.append(" ".join(jieba.lcut(title)))
            labels.append(line["tag"])
    train_x = sentences_to_vectors(sentences, model)
    train_y = label_to_label_index(labels)
    return train_x, train_y

#tag标签转化为类别标号
def label_to_label_index(labels):
    return [LABELS[y] for y in labels]

#文本向量化，使用了基于这些文本训练的词向量
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
                # vector = np.max([vector, model.wv[word]], axis=0)
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model("../上午/model.w2v")
    train_x, train_y = load_sentence("../上午/train_tag_news.json", model)
    test_x, test_y = load_sentence("../上午/valid_tag_news.json", model)
    #classifiers = [SVC(), DecisionTreeClassifier(), RandomForestClassifier()]
    func = [100, 200, 50]
    classifiers = [RandomForestClassifier(n_estimators=x) for x in func]
    for index, classifier in enumerate(classifiers):
        classifier.fit(train_x, train_y)
        y_pred = classifier.predict(test_x)
        print(classifier.__class__)
        print(classification_report(test_y, y_pred))



if __name__ == "__main__":
    main()