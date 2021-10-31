'''
作业：根据词典，输出一段文本所有可能的切割方式
'''


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

sentence = "经常有意见分歧"

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

# 由前缀词典构建有向无环图,由有向无环图来得到所有可能路径
def get_DAG(Dict, sentence):
        DAG = {}
        N = len(sentence)
        # 依次遍历文本中的每个位置
        for k in range(N):
                tmplist = []
                i = k
                # 位置k形成的片段
                frag = sentence[k]
                # 判断片段是否在前缀词典中,如果不在前缀词典中，则跳出本次循环,也即该片段已经超出统计词典的范围
                while i < N:
                        # 如果该片段在前缀词典中,将该片段加入到有向无环图当中,否则，继续循环
                        if frag in Dict:
                                tmplist.append(i)
                        i += 1
                        # 新的片段比旧的片段右边新增一个字
                        frag = sentence[k:i + 1]
                DAG[k] = tmplist
        return DAG
dag = get_DAG(Dict, sentence)
print(dag)

all = []
for key, value in dag.items():
    tmp = []
    for num in value:
        if key == num:
            tmp.append((key,))
        else:
            tmp.append((key, num))
    all.append(tmp)
print(all)

# 采用动态规划方法找到所有的数字组合，然后再索引得到结果，然后参考了同学的方法，觉得翟同学的写的更好。
# 摘录了下来

# 翟同学的动态规划方法
from pprint import pprint
tokens = Dict.keys() # dict_keys(['经常', '经', '有', '常', '有意见', '歧', '意见', '分歧', '见', '意', '见分歧', '分'])
res = []
def DFS(sentence, tokens, idx, subset, res):
    if idx == len(sentence):
        res.append(subset[:])
        return

    for w in tokens:
        subset.append(w)
        if sentence[idx:idx+len(w)] == w:
            DFS(sentence, tokens, idx+len(w), subset, res)
        subset.pop()

DFS(sentence, tokens, 0, [], res)
pprint(res)