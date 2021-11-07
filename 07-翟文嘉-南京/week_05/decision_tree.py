import math

'''
计算一个集合的信息熵
'''

#计算信息熵
def entropy_infomation(elements):
    ent = 0
    for element in set(elements):
        p = elements.count(element) / len(elements)
        ent += p * math.log(p, 2)
    return -ent

elements = [1, 1, 1, 1, 1, 1]
print(entropy_infomation(elements))
elements = [1, 1, 1, 2, 2, 2]
print(entropy_infomation(elements))
elements = [1, 2, 3, 4, 5, 6]
print(entropy_infomation(elements))
elements = [1, 1, 1, 1, 1, 2]
print(entropy_infomation(elements))

#计算信息增益
#离散变量
def entropy_gain_discrete(features, labels):
    #样本总信息熵Ent(D)
    ent_d = entropy_infomation(labels)
    for feature in set(features):
        #选出特征值=feature的样本
        feature_elements = [label for i, label in enumerate(labels) if features[i] == feature]
        #计算信息熵
        ent_d_feature = entropy_infomation(feature_elements)
        ent_d -= len(feature_elements) / len(labels) * ent_d_feature
    return ent_d

#计算信息增益
#连续变量
def entropy_gain_continuous(features, labels):
    ent_d = entropy_infomation(labels)
    #计算划分点
    splits = []
    uni_features = sorted(list(set(features)))
    for i in range(len(uni_features) - 1):
        splits.append((uni_features[i] + uni_features[i + 1]) / 2)
    gains = []
    for split in splits:
        #大于切分值的样本
        sample_bigger = [label for i, label in enumerate(labels) if features[i] >= split]
        #小于切分值的样本
        sample_smaller = [label for i, label in enumerate(labels) if features[i] < split]
        ent_bigger = entropy_infomation(sample_bigger)
        ent_smaller = entropy_infomation(sample_smaller)
        split_gains = ent_d - (len(sample_smaller) / len(labels) * ent_smaller
                               + len(sample_bigger) / len(labels) * ent_bigger)
        gains.append(split_gains)
    # 选择切分后信息增益最大的切分点
    gain = max(gains)   #最大增益值
    split = splits[gains.index(gain)] #最大切分点
    return gain, split

#”是否有房产“, 1 = 是, 0 = 否
feature_fangchan = [1,0,0,1,0,0,1,0,0,0]
#"婚姻状况", 结婚 = 0， 单身 = 1， 离婚 = 2
feature_hunyin = [1,0,1,0,2,0,2,1,0,1]
#“收入”  连续变量
feature_shouru = [125, 100, 70, 120, 95, 60, 220, 85, 75, 90]
#标签  “是否无法偿还” 0=否  1=是
labels = [0,0,0,0,1,0,0,1,0,1]

print(entropy_gain_discrete(feature_fangchan, labels))
print(entropy_gain_discrete(feature_hunyin, labels))
print(entropy_gain_continuous(feature_shouru, labels))