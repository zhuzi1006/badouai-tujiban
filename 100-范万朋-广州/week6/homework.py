# -*- codeing = utf-8 -*-
# @Time: 2021/11/9 20:46
# @Author: 棒棒朋
# @File: homework.py
# @Software: PyCharm
import copy
import math
from collections import defaultdict
import jieba

"""
第六周作业：文本纠错任务，使用N-gram算法计算成句概率

第8个字建议修改：平 -> 秤,概率提升：16.666266
修改前： 常见患病人群：天平座、天蟹座、双羽座
修改后： 常见患病人群：天秤座、天蟹座、双羽座

"""


class NgramLanguageModel:
    def __init__(self, corpus=None, n=3):
        self.n = n
        self.sep = "^"  # 用来分割两个词，没有实际含义，只要是字典里不存在的符号都可以
        self.sos = "<sos>"  # start of sentence，句子开始的标识符
        self.eos = "<eos>"  # end of sentence，句子结束的标识符
        self.unk_prob = 1e-5  # 给unk分配一个比较小的概率值，避免集外词概率为0
        self.fix_backoff_prob = 0.4  # 使用固定的回退概率
        self.ngram_count_dict = dict((x + 1, defaultdict(int)) for x in range(n))
        self.ngram_count_prob_dict = dict((x + 1, defaultdict(int)) for x in range(n))
        self.ngram_count(corpus)
        self.calc_ngram_prob()

    # 将文本切分成词或字或token
    def sentence_segment(self, sentence):
        return jieba.lcut(sentence)

    # 统计ngram的数量
    def ngram_count(self, corpus):
        for sentence in corpus:
            word_lists = self.sentence_segment(sentence)
            word_lists = [self.sos] + word_lists + [self.eos]  # 前后补充开始符和结尾符
            for window_size in range(1, self.n + 1):  # 按不同窗长扫描文本
                for index, word in enumerate(word_lists):
                    # 取到末尾时窗口长度会小于指定的gram，跳过那几个
                    if len(word_lists[index:index + window_size]) != window_size:
                        continue
                    # 用分隔符连接word形成一个ngram用于存储
                    ngram = self.sep.join(word_lists[index:index + window_size])
                    self.ngram_count_dict[window_size][ngram] += 1
        # 计算总词数，后续用于计算一阶ngram概率
        self.ngram_count_dict[0] = sum(self.ngram_count_dict[1].values())
        return

    # 计算ngram概率
    def calc_ngram_prob(self):
        for window_size in range(1, self.n + 1):
            for ngram, count in self.ngram_count_dict[window_size].items():
                if window_size > 1:
                    ngram_splits = ngram.split(self.sep)  # ngram        :a b c
                    ngram_prefix = self.sep.join(ngram_splits[:-1])  # ngram_prefix :a b
                    ngram_prefix_count = self.ngram_count_dict[window_size - 1][ngram_prefix]  # Count(a,b)
                else:
                    ngram_prefix_count = self.ngram_count_dict[0]  # count(total word)

                self.ngram_count_prob_dict[window_size][ngram] = count / ngram_prefix_count

        return

    # 获取ngram概率，其中用到了回退平滑，回退概率采取固定值
    def get_ngram_prob(self, ngram):
        n = len(ngram.split(self.sep))
        if ngram in self.ngram_count_prob_dict[n]:
            # 尝试直接取出概率
            return self.ngram_count_prob_dict[n][ngram]
        elif n == 1:
            # 一阶gram查找不到，说明是集外词，不做回退
            return self.unk_prob
        else:
            # 高于一阶的可以回退
            ngram = self.sep.join(ngram.split(self.sep)[1:])
            return self.fix_backoff_prob * self.get_ngram_prob(ngram)

    # 回退法预测句子概率
    def predict(self, sentence):
        word_list = self.sentence_segment(sentence)
        word_list = [self.sos] + word_list + [self.eos]
        sentence_prob = 0
        for index, word in enumerate(word_list):
            ngram = self.sep.join(word_list[max(0, index - self.n + 1):index + 1])
            prob = self.get_ngram_prob(ngram)
            sentence_prob += math.log(prob)
        return sentence_prob


class Corrector:
    def __init__(self, language_model):
        self.language_model = language_model
        # 候选字字典
        self.sub_dict = self.load_tongyinzi("tongyin.txt")
        # 成句概率的提升超过阈值则保留修改
        self.threshold = 10
        # 一个原句的成句概率，当做阈值使用
        self.sentence_prob_baseline = 0

    def load_tongyinzi(self, path):
        """
        加载同音字字典表，同形字等也可以加入，本质上是常用的错字
        """
        tongyinzi_dict = {}
        with open(path, encoding="utf8") as f:
            for line in f:
                char, tongyin_chars = line.split()
                tongyinzi_dict[char] = list(tongyin_chars)
        return tongyinzi_dict

    def get_candidate_sentence_prob(self, candidates, char_list, index):
        """
        根据【替换字】逐句计算成句概率的提升值
        :param candidates: 该字的【替换字】候选列表
        :param char_list: 将句子按字为单位切割
        :param index: 需要替换的当前字的索引
        """
        if candidates == []:
            return [-1]
        result = []
        for char in candidates:
            char_list[index] = char
            sentence = "".join(char_list)

            sentence_prob = self.language_model.predict(sentence)
            # 减去基线值，得到提升了多少
            sentence_prob -= self.sentence_prob_baseline
            result.append(sentence_prob)
        return result

    # 纠错逻辑
    def correction(self, string):
        """
        ①先计算原始句子的成句概率
        ②拿到需要纠错的句子之后，将其按字分割放到列表中
        ③循环句子中的每个字,拿到这个字的同音字集合
        ④循环这个字的同音字列表中的每个词，把每个词都替换到原始句子中，计算此刻新句子的成句概率
        ⑤计算新的成句概率是否大于一定阈值，大于则用新的同音字进行替换
        """
        char_list = list(string)
        fix = {}
        # 计算一个原句的成句概率
        self.sentence_prob_baseline = self.language_model.predict(string)
        for index, char in enumerate(char_list):
            candidates = self.sub_dict.get(char, [])  # 获取该字char 的同音字集合
            # 注意使用char_list的拷贝，以免直接修改了原始内容
            candidate_probs = self.get_candidate_sentence_prob(candidates, copy.deepcopy(char_list), index)
            # 如果成句概率的提升大于一定阈值，则记录替换结果

            if max(candidate_probs) > self.threshold:
                # 找到最大成句概率对于的替换字
                sub_char = candidates[candidate_probs.index(max(candidate_probs))]
                print("第%d个字建议修改：%s -> %s,概率提升：%f" % (index, char, sub_char, max(candidate_probs)))
                fix[index] = sub_char
        # 替换后的字符串
        char_list = [fix[i] if i in fix else char for i, char in enumerate(char_list)]
        return "".join(char_list)


if __name__ == '__main__':
    corpus = open("corpus/星座.txt", encoding="utf8").readlines()
    lm = NgramLanguageModel(corpus, 3)

    cr = Corrector(lm)
    string = "常见患病人群：天平座、天蟹座、双羽座"
    fix_string = cr.correction(string)
    print("修改前：", string)
    print("修改后：", fix_string)
