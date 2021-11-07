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

sentence = "经常有意见分歧"

# 提取 keys，数值用不到
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