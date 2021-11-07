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

① 使用测试数据集 valid_tag_news.json
② 加载标题、内容
③ 数据预处理，替换掉字符'_'为'-'

预测结果：
    属于类别[国际]的概率是0.216396
    属于类别[财经]的概率是0.109361
    属于类别[军事]的概率是0.084037
    属于类别[科技]的概率是0.040476
    属于类别[房产]的概率是0.016396
"""

import jieba
import json
from collections import defaultdict

# import jieba 时不会立刻加载jieba词典，使用时才开始加载。如果想提前加载和初始化，可以手动触发
jieba.initialize()


def load_data(path):
    """
    加载数据
    :param path: json数据路径
    :return: corpus:所有标签（18个）包含的句子切词列表
    :return: all_words:所有词（6815个）
    :return: all_class:所有类别标签（18个）
    """
    all_class = {}
    all_words = set()
    corpus = defaultdict(list)
    with open(path, encoding="utf8") as f:
        for line in f:
            line = json.loads(line)
            tag = line["tag"]  # 获取所有标签，共18个
            all_class[tag] = len(all_class)
            title, content = line["title"], line["content"]  # 获取标题,获取内容

            all_text = title + content  # 训练标题和内容
            all_text = all_text.replace("_", "-")  # 替换掉字符'_'为'-'
            words = jieba.lcut(all_text)  # 切割标题+内容
            # union() 方法返回两个/多个集合的并集，重复的元素只会出现一次。
            all_words = all_words.union(set(words))  # 所有的词
            corpus[tag].append(words)  # 把切分好的词放到对应标签里面
    return corpus, all_words, all_class


def calculate_p_class(corpus):
    """
    计算每类别的概率 P(class_x)
    从训练数据中直接计算，有多少样本属于class_x,除以样本总数
    :param corpus: 每个类别的语料
    :return: p_to_p_class：每个类别的概率
    """
    total = sum([len(data) for data in corpus.values()])  # 计算总共有多少个样本
    p_to_p_class = {}
    for class_, class_data in corpus.items():
        # 用这个类别的样本个数 / 总样本个数 = 该类别的概率
        p_to_p_class[class_] = len(class_data) / total
    return p_to_p_class


def calculate_p_feature_class(corpus):
    """
    计算类别 x中，词 w出现的概率  P(feature_w|class_x)
    :param corpus: 每个类别切分好语料
    :return: 该词在该类别中的概率 = 出现过该词的样本/该类别样本总数
    """
    feature_class_to_p_feature_class = {}
    for class_, word_lists in corpus.items():  # 循环每个类别:文化、时尚....
        word_appearance_dict = defaultdict(set)  # 记录每个词在哪些样本中出现过
        for index, word_list in enumerate(word_lists):  # 训练每个类别中每个句子的切词列表
            for word in word_list:
                word_appearance_dict[word].add(index)
        for word, word_appearance in word_appearance_dict.items():
            key = word + "_" + class_  # 用词+类别的方式作为key “你好_文化、不_文化”
            # 计算：该词在该类别中的概率 = 出现过该词的样本/该类别样本总数
            feature_class_to_p_feature_class[key] = len(word_appearance) / len(word_lists)
    return feature_class_to_p_feature_class


def calculate_p_feature(class_to_p_class, feature_class_to_p_feature_class, all_words):
    """
    计算每个词的出现概率 P(feature_w)，利用全概率公式
    P(feature_w) =
        P(feature_w|class1)*P(class1) + P(feature_w|class2)*P(class2) + ... + P(feature_w|classn)*P(classn)

    :param class_to_p_class: 每类别的概率 P(class_x)
    :param feature_class_to_p_feature_class: 类别 x中，词 w出现的概率  P(feature_w|class_x)
    :param all_words: 所有的词（已切好）
    :return: 每一个词出现的概率
    """
    feature_to_p_feature = {}
    for word in all_words:  # 遍历每一个词
        prob = 0
        # 全概率公式
        for class_, p_class in class_to_p_class.items():  # 这个词在每个类别中遍历
            key = word + "_" + class_
            prob += p_class * feature_class_to_p_feature_class.get(key, 0)  # 计算这个词在所有类别中的概率
        feature_to_p_feature[word] = prob  # 将这个词的概率存储起来
    return feature_to_p_feature


def bayes_algorithm(corpus, all_words):
    """
    贝叶斯算法  P(A|B) = (P(A) * P(B|A)) / P(B)
    我们要求解的目标:
            P(class_x|feature_w) =
                                (P(class_x) * P(feature_w|class_x)) / P(feature_w)
    :param corpus: corpus是经过处理的语料数据
        形式是以类别序号为key，值是每个样本中word组成的
        {0:[["体育","新闻"]...], 1:[["财经","新闻"]...]}
        部分非重要词被替换为默认token
    :param all_words: 所有的词（已切好）
    :return:
    """
    # 记录所有的P(class_x|feature_w)
    bayes_feature_dict = {}
    # {'文化': 0.058, '时尚': 0.056, '健康': 0.054,....., '房产': 0.051, '社会': 0.054}
    class_to_p_class = calculate_p_class(corpus)  # 求解出P(class_x)
    # {'“_文化': 0.30, '少年_文化': 0.01,......, '最小_社会': 0.01, '蛇种_社会': 0.01}
    feature_w_class_x = calculate_p_feature_class(corpus)  # 求解 P(feature_w|class_x)
    # 求解 P(feature_w) {"你好":0.3，"再见":0.1...}
    feature_to_p_feature = calculate_p_feature(class_to_p_class, feature_w_class_x, all_words)
    for feature_class, p_feature_class in feature_w_class_x.items():
        feature, class_ = feature_class.split("_")
        p_class = class_to_p_class[class_]  # 取出 P(class_x)
        p_feature = feature_to_p_feature[feature]  # 取出 P(feature_w)
        # P(class_x|feature_w) = (P(class_x) * P(feature_w|class_x)) / P(feature_w)
        bayes_feature_dict[feature_class] = (p_class * p_feature_class) / p_feature
    return bayes_feature_dict


def bayes_predict(query, bayes_feature_dict, all_class, topk=5):
    """
    预测环节
    :param query: 文本句子
    :param bayes_feature_dict: 贝叶斯预测出来的字典：每一个词出现在每一个类别中的概率
    :param all_class: 所有的类别标签（18个）
    :param topk: 取出前 k 个可能的类别概率
    :return:
    """
    words = jieba.lcut(query)
    records = []
    for class_ in all_class:  # 对某个词遍历所有类别
        p_class = 0  # 初始化该词在每个类中的概率
        for word in words:  # 遍历该句子的所有词
            default_value = 0  # 如果该词不在该类中，那就默认为0
            p = bayes_feature_dict.get(word + "_" + class_, default_value)
            p_class += p  # 累加的结果：该句子属于该类别的概率
        records.append([class_, p_class / len(words)])

    for class_, prob in sorted(records, reverse=True, key=lambda x: x[1])[:topk]:
        print("属于类别[%s]的概率是%f" % (class_, prob))


if __name__ == "__main__":
    path = "valid_tag_news.json"
    corpus, all_words, all_class = load_data(path)

    bayes_feature_dict = bayes_algorithm(corpus, all_words)  # 字典：每一个词在每一个类别中的概率
    query = "菲律宾向越南示好归还所扣7艘越方渔船"
    bayes_predict(query, bayes_feature_dict, all_class)
