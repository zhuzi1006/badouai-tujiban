import math
from collections import defaultdict
import jieba


class NgramLanguageModel:
    def __init__(self, corpus=None, n=3):
        self.n = n
        self.sep = "_"     # 用来分割两个词，没有实际含义，只要是字典里不存在的符号都可以
        self.sos = "<sos>"    #start of sentence，句子开始的标识符
        self.eos = "<eos>"    #end of sentence，句子结束的标识符
        self.unk_prob = 1e-5  #给unk分配一个比较小的概率值，避免集外词概率为0
        self.fix_backoff_prob = 0.4  #使用固定的回退概率
        self.ngram_count_dict = dict((x + 1, defaultdict(int)) for x in range(n))
        self.ngram_count_prob_dict = dict((x + 1, defaultdict(int)) for x in range(n))
        self.ngram_count(corpus)
        self.calc_ngram_prob()

    #将文本切分成词或字或token
    def sentence_segment(self, sentence):
        # return sentence.split()
        return jieba.lcut(sentence)

    #统计ngram的数量
    def ngram_count(self, corpus):
        for sentence in corpus:
            sentence=sentence.replace(" ","")
            sentence = sentence.replace('_', '')
            word_lists = self.sentence_segment(sentence)

            word_lists = [self.sos] + word_lists + [self.eos]  #前后补充开始符和结尾符
            for window_size in range(1, self.n + 1):           #按不同窗长扫描文本
                for index, word in enumerate(word_lists):
                    #取到末尾时窗口长度会小于指定的gram，跳过那几个
                    if len(word_lists[index:index + window_size]) != window_size:
                        continue
                    #用分隔符连接word形成一个ngram用于存储
                    ngram = self.sep.join(word_lists[index:index + window_size])
                    if(ngram=='已有__'):
                        str=''
                    self.ngram_count_dict[window_size][ngram] += 1
        #计算总词数，后续用于计算一阶ngram概率
        self.ngram_count_dict[0] = sum(self.ngram_count_dict[1].values())
        return

    #计算ngram概率
    def calc_ngram_prob(self):
        for window_size in range(1, self.n + 1):
            for ngram, count in self.ngram_count_dict[window_size].items():
                if window_size > 1:
                    ngram_splits = ngram.split(self.sep)              #ngram        :a b c
                    ngram_prefix = self.sep.join(ngram_splits[:-1])   #ngram_prefix :a b
                    ngram_prefix_count = self.ngram_count_dict[window_size - 1][ngram_prefix] #Count(a,b)
                else:
                    ngram_prefix_count = self.ngram_count_dict[0]     #count(total word)
                # word = ngram_splits[-1]
                # self.ngram_count_prob_dict[word + "|" + ngram_prefix] = count / ngram_prefix_count
                self.ngram_count_prob_dict[window_size][ngram] = count / ngram_prefix_count
        return

    #获取ngram概率，其中用到了回退平滑，回退概率采取固定值
    def get_ngram_prob(self, ngram):
        n = len(ngram.split(self.sep))
        if ngram in self.ngram_count_prob_dict[n]:
            #尝试直接取出概率
            return self.ngram_count_prob_dict[n][ngram]
        elif n == 1:
            #一阶gram查找不到，说明是集外词，不做回退
            return self.unk_prob
        else:
            #高于一阶的可以回退
            ngram = self.sep.join(ngram.split(self.sep)[1:])
            return self.fix_backoff_prob * self.get_ngram_prob(ngram)


    #回退法预测句子概率
    def calc_sentence_prob(self, sentence):
        word_list = self.sentence_segment(sentence)
        word_list = [self.sos] + word_list + [self.eos]
        sentence_prob = 0
        for index, word in enumerate(word_list):
            ngram = self.sep.join(word_list[max(0, index - self.n + 1):index + 1])
            prob = self.get_ngram_prob(ngram)
            # print(ngram, prob)
            sentence_prob += math.log(prob)
        return sentence_prob



def Correr(str,lm):
    words = open("tongyi.txt", encoding="utf8").readlines()
    tong_dict={}
    threshold=10
    # 加载同音字
    for sentence in words:
        item1=sentence.split(' ')[0]
        item2=sentence.split(' ')[1]
        tong_dict[item1]=list(item2)
    lst_str=list(str)
    rep_dict={}
    for i in range(len(lst_str)):
        o1 = lm.calc_sentence_prob(str)
        if(tong_dict.get(str[i])!=None):
            item2=tong_dict.get(str[i])
            coll_f={}
            for c in item2:
                lst_copy=lst_str.copy()
                lst_copy[i]=c
                o2=lm.calc_sentence_prob(''.join(lst_copy))
                # 如果替换后的同音字成句概率超过阈值，则记录
                if(o2-o1>threshold):
                    coll_f[c]=(o2-o1)
            if(coll_f.__len__()>0):
                # 在多个同音字中选取值最大的同音字
                ti = max(zip(coll_f.values(), coll_f.keys()))[1]
                v = max(zip(coll_f.values(), coll_f.keys()))[0]
                print("建议将原句中", str[i], '替换为', ti, "提升", v)
                rep_dict[i] = ti


    print("原句:",str," 成句概率:",lm.calc_sentence_prob(str))
    for key in rep_dict:
        lst_str[key]=rep_dict[key]
    rep_str=''.join(lst_str)
    print("替换后:",rep_str, " 成句概率:" , lm.calc_sentence_prob(rep_str))


if __name__ == "__main__":
    corpus = open("财经.txt", encoding="utf8").readlines()
    lm = NgramLanguageModel(corpus, 3)
    str='节后恺市第一轴'
    Correr(str,lm)

