#!/usr/bin/env python3  
# coding: utf-8

"""
# 使用基于词向量的分类器
# 对比几种模型的效果

调参设置：
SVC(kernel='linear', gamma='auto')
准确率更正情况： 44% --> 51%

DecisionTreeClassifier(criterion='entropy', splitter='best', min_samples_split=4)
准确率更正情况： 31% --> 34%

RandomForestClassifier(criterion='entropy', n_estimators=150)
准确率更正情况： 48% --> 49%

"""
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict

LABELS = {'健康': 0, '军事': 1, '房产': 2, '社会': 3, '国际': 4, '旅游': 5,
          '彩票': 6, '时尚': 7, '文化': 8, '汽车': 9, '体育': 10, '家居': 11,
          '教育': 12, '娱乐': 13, '科技': 14, '股票': 15, '游戏': 16, '财经': 17}


# 输入模型文件路径
# 加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


def load_sentence(path, model):
    """
    加载数据集
    :param path: json数据集路径
    :param model: 词向量模型
    :return: 训练数据，训练标签
    """
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


# tag标签转化为类别标号
def label_to_label_index(labels):
    return [LABELS[y] for y in labels]


# 文本向量化，使用了基于这些文本训练的词向量
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
    model = load_word2vec_model("model.w2v")
    train_x, train_y = load_sentence("train_tag_news.json", model)
    test_x, test_y = load_sentence("valid_tag_news.json", model)
    classifiers = [SVC(kernel='linear', gamma='auto'),
                   DecisionTreeClassifier(criterion='entropy', splitter='best', min_samples_split=4),
                   RandomForestClassifier(criterion='entropy', n_estimators=150)]
    # func = [100, 200, 50]
    # classifiers = [RandomForestClassifier(n_estimators=x) for x in func]
    for index, classifier in enumerate(classifiers):
        classifier.fit(train_x, train_y)
        y_pred = classifier.predict(test_x)
        print(classifier.__class__)
        '''
        TP: 正类预测  正类标签
        FP: 正类预测  负类标签
        TN: 负类预测  负类标签
        FN: 负类预测  正类标签
        准确率/精度(precision) = 正确预测的个数/被预测正确的个数即：TP/(TP+FP)
        召回率(recall)=正确预测的个数/预测个数即：TP/(TP + FN)
        F1-score= 2*精度*召回率/(精度+召回率)
        '''
        print(classification_report(test_y, y_pred))


if __name__ == "__main__":
    main()
