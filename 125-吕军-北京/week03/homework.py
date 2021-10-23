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


def cut_method(string, word_dict):
        def sentences(cur):
                result = []
                if cur < len(string):
                        for next in range(cur + 1, len(string) + 1):
                                if (string[cur:next] in word_dict):
                                        result = result + [string[cur:next] + (tail and ',' + tail) for tail in
                                                           sentences(next)]
                else:
                        return ['']
                return result

        words = []
        for line in sentences(0):
                line = line.split(',')
                words.append(line)

        return words

if __name__ == '__main__':
    print(cut_method(sentence, Dict))

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

