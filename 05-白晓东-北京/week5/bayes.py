
import jieba
import re
import os
import json
from collections import defaultdict

jieba.initialize()

"""
贝叶斯分类实践

P(A|B) = (P(A) * P(B|A)) / P(B)
事件A：文本属于类别x。文本属于类别x的概率，记做P(class_x)
事件B：文本包含词w。文本包含词w的概率，记做P(feature_w)
P(class_x|feature_w) = (P(class_x) * P(feature_w|class_x)) / P(feature_w)
P(class_x|feature_w)是我们要求解的目标
P(class_x)从训练数据中直接计算，有多少样本属于class_x,除以样本总数
P(feature_w|class_x)从训练数据中计算，有多少属于class_x的样本，包含词w
P(feature_w)的计算，依靠全概率公式：  
    P(feature_w) = P(feature_w|class1)*P(class1) + P(feature_w|class2)*P(class2) + ... + P(feature_w|classn)*P(classn)
"""

def load_data(path):
    all_class = {}
    all_words = set()
    corpus = defaultdict(list)
    with open(path, encoding="utf8") as f:
        for line in f:
            line = json.loads(line)
            tag = line["tag"]
            all_class[tag] = len(all_class)
            title = line["title"]
            words = jieba.lcut(title)
            all_words = all_words.union(set(words))
            corpus[tag].append(words)
    return corpus, all_words, all_class

#计算每类别的概率
def calculate_p_class(corpus):
    total = sum([len(data) for data in corpus.values()])  # 一共有多少句话
    p_to_p_class = {}
    for class_, class_data in corpus.items():
        p_to_p_class[class_] = len(class_data) / total
    return p_to_p_class

#计算类别x中，词w出现的概率
def calculate_p_feature_class(corpus):
    feature_class_to_p_feature_class = {}
    for class_, word_lists in corpus.items():
        word_appearance_dict = defaultdict(set) #记录每个词在哪些样本中出现过
        for index, word_list in enumerate(word_lists):
            for word in word_list:
                word_appearance_dict[word].add(index)
        for word, word_appearance in word_appearance_dict.items():
            key = word + "_" + class_ #用词+类别的方式作为key “你好_0”
            #出现过该词的样本/该类别样本总数
            feature_class_to_p_feature_class[key] = len(word_appearance) / len(word_lists)
    return feature_class_to_p_feature_class


#计算每个词的出现概率，需要利用全概率公式
def calculate_p_feature(class_to_p_class, feature_class_to_p_feature_class, all_words):
    feature_to_p_feature = {}
    for word in all_words:
        prob = 0
        #全概率公式
        for class_, p_class in class_to_p_class.items():
            key = word + "_" + class_
            prob += p_class * feature_class_to_p_feature_class.get(key, 0)
        feature_to_p_feature[word] = prob
    return feature_to_p_feature

#corpus是经过处理的语料数据
#形式是以类别序号为key，值是每个样本中word组成的
#{0:[["体育","新闻"]...], 1:[["财经","新闻"]...]}
#部分非重要词被替换为默认token
def bayes_algorithm(corpus, all_words):
    # 记录所有的P(class_x|feature_w)
    bayes_feature_dict = {}
    # {0:0.1, 1:0.2, 2:0.3....}
    class_to_p_class = calculate_p_class(corpus)
    # {"你好_0":0.1, "你好_1":0.2...}
    feature_class_to_p_feature_class = calculate_p_feature_class(corpus)
    # {"你好":0.3，"再见":0.1...}
    feature_to_p_feature = calculate_p_feature(class_to_p_class, feature_class_to_p_feature_class, all_words)
    for feature_class, p_feature_class in feature_class_to_p_feature_class.items():
        feature, class_ = feature_class.split("_")
        p_class = class_to_p_class[class_]
        p_feature = feature_to_p_feature[feature]
        # P(class_x|feature_w) = (P(class_x) * P(feature_w|class_x)) / P(feature_w)
        bayes_feature_dict[feature_class] = (p_class * p_feature_class) / p_feature
    return bayes_feature_dict

#预测环节
def bayes_predict(query, bayes_feature_dict, all_class, topk=5):
    words = jieba.lcut(query)
    records = []
    for class_ in all_class:
        p_class = 0
        for word in words:
            default_value = 0
            p = bayes_feature_dict.get(word + "_" + class_, default_value)
            p_class += p
        records.append([class_, p_class/len(words)])
    records_sorted = sorted(records, reverse=True, key=lambda x:x[1])
    for class_, prob in records_sorted[:topk]:
        print("%s。这个新闻标题属于类别[%s]的概率是%f"%(query, class_, prob))
    print('*************************')
    return records_sorted


# 模型评价
def evaluate(valid_coprus_path, bayes_feature_dict, all_class):
    with open(valid_coprus_path, encoding='utf-8') as f:
        # 统计各个类别中所有的title，正确和错误的各占有多少。
        accuracy_dict = defaultdict(list)
        # 各个类别的正确率
        accuracy = defaultdict(int)
        title_num = 0
        for line in f:
            line = json.loads(line)
            tag = line['tag']
            if tag not in accuracy_dict.keys():
                accuracy_dict[tag].extend([0, 0])
            title = line['title']
            records = bayes_predict(title, bayes_feature_dict, all_class)
            if records[0][0] == tag:
                accuracy_dict[tag][0] += 1
            else:
                accuracy_dict[tag][1] += 1
            title_num += 1
        print("一共测试了{}个句子".format(title_num))
        for class_, value in accuracy_dict.items():
            accuracy[class_] = round((value[0] / (value[0] + value[1])), 4)
        print('各个类别的准确率统计如下:{}'.format(accuracy))
        score = 0
        for value in accuracy.values():
            score += value
        score /= len(accuracy)
        print('在该语料上，总的准确率为：%.4f' % score)
    return

if __name__ == "__main__":
    path = "train_tag_news.json"
    corpus, all_words, all_class = load_data(path)
    bayes_feature_dict = bayes_algorithm(corpus, all_words)
    query = "2021年12月31日，我国武统台湾，实现了祖国的统一。"
    bayes_predict(query, bayes_feature_dict, all_class)
    # 模型评价
    evaluate('tag_news.json', bayes_feature_dict, all_class)


