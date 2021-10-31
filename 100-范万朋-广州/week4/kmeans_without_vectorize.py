# coding: utf-8
"""
实现基于字符串的Kmeans算法
不需要对文本向量化，直接使用字符串匹配算法进行聚类
"""
import random
import sys
import json


class KMeansCluster:
    def __init__(self, sentences, cluster_num):
        self.sentences = sentences
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(sentences, cluster_num)
        self.buffer = {}  # 因为过程中会有许多重复的距离计算，做一个缓存字典

    def cluster(self):
        result = []
        for i in range(self.cluster_num):
            result.append([])
        for item in self.sentences:
            distance_min = sys.maxsize
            index = -1
            for i in range(len(self.points)):
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item]
        new_center = []
        distances = []
        for item in result:
            center, distance_to_all = self.__center(item)
            new_center.append(center)
            distances.append(distance_to_all)
        # 中心点未改变，说明达到稳态，结束递归
        if self.points == new_center:
            return result

        self.points = new_center
        return self.cluster()

    # 选取新的中心的方法
    # 由于使用的是离散的字符串，所以无法通过原有的平均方式计算新的中心
    # 人为设定新中心更替方式：
    # 选取类别中到其他所有点距离总和最短的字符串为中心
    def __center(self, cluster_sentences):
        center = "             "  # 设置一个不存在的占位字符
        distance_to_all = 999999999  # 占位最大距离
        for sentence_a in cluster_sentences:
            distance = 0
            for sentence_b in cluster_sentences:
                distance += self.__distance(sentence_a, sentence_b)
            distance /= len(sentences)
            if distance < distance_to_all:
                center = sentence_a
                distance_to_all = distance
        return center, distance_to_all

    # 将距离函数替换为非向量算法
    # 此处使用jaccard距离
    # 使用字典缓存加快距离计算
    def __distance(self, p1, p2):
        if p1 + p2 in self.buffer:
            return self.buffer[p1 + p2]
        elif p2 + p1 in self.buffer:
            return self.buffer[p2 + p1]
        else:
            # jaccard(杰卡德)距离：公共词越多越相近
            distance = 1 - len(set(p1) & set(p2)) / len(set(p1).union(set(p2)))
            self.buffer[p1 + p2] = distance
            return distance

    # 随机选取初始点,改成随机挑选字符串
    def __pick_start_point(self, sentences, cluster_num):
        return random.sample(sentences, cluster_num)


# 加载数据集
def load_sentence(path):
    sentences = []
    with open(path, encoding="utf8") as f:
        for index, line in enumerate(f):
            sentences.append(line.strip())
    return sentences


if __name__ == '__main__':
    sentences = load_sentence("titles.txt")
    km = KMeansCluster(sentences, 42)
    res = km.cluster()
    kmeans_json = json.dumps(res, ensure_ascii=False, indent=2)

    print(kmeans_json)
