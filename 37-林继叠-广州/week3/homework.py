'''
作业：根据词典，输出一段文本所有可能的切割方式
'''


#词典，每个词后方存储的是其词频，仅为示例，也可自行添加
from copy import deepcopy

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

len_setence = len(sentence)
# 找到最长的单词
max_long_word = max([len(i) for i in Dict])
print(max_long_word)

# 全切分，找到每个词的切片及其相关的定位
word_slice_dict = dict()
for i in range(max_long_word):
        for j,dd in enumerate(zip(*[sentence[l:] for l in range(i+1)])):
                word = ''.join(dd)
                if word in Dict:
                        data = (word,len(word)+j)
                        if j in word_slice_dict:
                                word_slice_dict[j].append(data)
                        else:
                                word_slice_dict[j] = [data]

# 定义一个递归函数来对切分好的词进行组合
print(word_slice_dict)
device_list = []
def zuhe(index, word_list):
        for w in word_slice_dict[index]:
                new_word_list = deepcopy(word_list)
                new_word_list.append(w[0])
                if w[1] == len_setence:
                        device_list.append(new_word_list)
                        return
                else:
                        zuhe(w[1],new_word_list)
# 开始组合
for i in word_slice_dict[0]:
        zuhe(i[1],[i[0]])

print(device_list)




"""
输出
[['经', '常', '有', '意', '见', '分', '歧'], 
['经', '常', '有', '意', '见', '分歧'],
['经', '常', '有', '意', '见分歧'],
['经', '常', '有', '意见', '分', '歧'],
['经', '常', '有', '意见', '分歧'],
['经', '常', '有意见', '分', '歧'],
['经', '常', '有意见', '分歧'],
['经常', '有', '意', '见', '分', '歧'],
['经常', '有', '意', '见', '分歧'],
['经常', '有', '意', '见分歧'],
['经常', '有', '意见', '分', '歧'],
['经常', '有', '意见', '分歧'],
['经常', '有意见', '分', '歧'],
['经常', '有意见', '分歧']]
"""
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

