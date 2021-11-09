'''
作业：根据词典，输出一段文本所有可能的切割方式
'''
from pprint import pprint

#词典，每个词后方存储的是其词频，仅为示例，也可自行添加
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

def calc_dag(sentence):
    DAG = {} #DAG空字典，用来存储DAG有向无环图
    N = len(sentence)
    for k in range(N):
        tmplist = []
        i = k
        frag = sentence[k]
        while i < N:
            if frag in Dict:
                tmplist.append(i)
            i += 1
            frag = sentence[k:i + 1]
        if not tmplist:
            tmplist.append(k)
        DAG[k] = tmplist
    return DAG

sentence = "经常有意见分歧"
print(calc_dag(sentence))

class DAGDecode:
    #通过两个队列实现
    def __init__(self,sentence):
        self.sentence = sentence
        self.DAG = calc_dag(sentence)
        self.length = len(sentence)
        self.unfinish_path =[[]] #保存带待解码序列的队列
        self.finish_path = []  #保存解码完成的序列的队列

    def decode_next(self, path):
        path_length = len("".join(path))
        if path_length == self.length:  # 已完成解码
            self.finish_path.append(path)
            return
        candidates = self.DAG[path_length]
        new_paths = []
        for candidate in candidates:
            new_paths.append(path + [self.sentence[path_length:candidate + 1]])
        self.unfinish_path += new_paths  # 放入待解码对列
        return

    # 递归调用序列解码过程
    def decode(self):
        while self.unfinish_path != []:
            path = self.unfinish_path.pop()  # 从待解码队列中取出一个序列
            self.decode_next(path)  # 使用该序列进行解码

sentence = "经常有意见分歧"
dd = DAGDecode(sentence)
dd.decode()
print(dd.finish_path)

"""
预期输出
[['经常', '有意见', '分歧'], 
 ['经常', '有意见', '分', '歧'],
 ['经常', '有', '意见', '分歧'], 
 ['经常', '有', '意见', '分', '歧'], 
 ['经常', '有', '意', '见分歧'], 
 ['经常', '有', '意', '见', '分歧'], 
 ['经常', '有', '意', '见', '分', '歧'], 
 ['经', '常', '有意见', '分歧'], 
 ['经', '常', '有意见', '分', '歧'], 
 ['经', '常', '有', '意见', '分歧'], 
 ['经', '常', '有', '意见', '分', '歧'], 
 ['经', '常', '有', '意', '见分歧'], 
 ['经', '常', '有', '意', '见', '分歧'], 
 ['经', '常', '有', '意', '见', '分', '歧']]
"""

###其他思路

tokens = Dict.keys()  # dict_keys(['经常', '经', '有', '常', '有意见', '歧', '意见', '分歧', '见', '意', '见分歧', '分'])
res = []
def DFS(sentence, tokens, idx, subset, res):
    if idx == len(sentence):
        res.append(subset[:])
        return

    for w in tokens:
        subset.append(w)
        if sentence[idx:idx + len(w)] == w:
            DFS(sentence, tokens, idx + len(w), subset, res)
        subset.pop()


DFS(sentence, tokens, 0, [], res)

pprint(res)
