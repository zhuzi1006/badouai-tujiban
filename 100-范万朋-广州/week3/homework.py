"""
作业：根据词典，输出一段文本所有可能的切割方式

预期输出：                                    已实现的情况：
[['经常', '有意见', '分歧'],                   ①最大正向匹配---窗口滑动
 ['经常', '有意见', '分', '歧'],
 ['经常', '有', '意见', '分歧'],
 ['经常', '有', '意见', '分', '歧'],           ⑥神经网络
 ['经常', '有', '意', '见分歧'],               ③最大反向匹配---窗口滑动
 ['经常', '有', '意', '见', '分歧'],
 ['经常', '有', '意', '见', '分', '歧'],
 ['经', '常', '有意见', '分歧'],
 ['经', '常', '有意见', '分', '歧'],
 ['经', '常', '有', '意见', '分歧'],
 ['经', '常', '有', '意见', '分', '歧'],
 ['经', '常', '有', '意', '见分歧'],
 ['经', '常', '有', '意', '见', '分歧'],
 ['经', '常', '有', '意', '见', '分', '歧']]   ⑦字符串拆解
 其他输出：
 ['经常', '有意见', '分歧']                    ②最大正向匹配---前缀词典
 ['经常', '有', '意见分歧']                    ④结巴分词
 ['经常', '有意', '见分歧']                    ⑤结巴分词---加载自定义字典
"""
import jieba
import RNN_cut_word


# ①最大正向匹配---窗口滑动
def forward_max_match_method1(text, dict_path, max_len):
    """
    正向最大匹配 方法1，窗口滑动
    :param text: 输入字符串
    :param dict_path: 字典路径
    :param max_len: 滑动窗口的最大长度
    :return: 切词后的列表
    """
    words = []
    word_dict = {}
    with open(dict_path, encoding="utf8") as file:
        # 打开txt路径，构建字典
        for line in file:
            word = line.split()[0]
            word_dict[word] = 0
    while text != '':
        lens = min(max_len, len(text))
        word = text[:lens]
        while word not in word_dict:
            if len(word) == 1:
                break
            word = word[:len(word) - 1]
        words.append(word)
        text = text[len(word):]
    return words


# ②最大正向匹配---前缀词典
def load_prefix_word_dict(path):
    """
    加载词前缀词典
    # 用0和1来区分是前缀还是真词
    # 需要注意有的词的前缀也是真词，在记录时不要互相覆盖
    :param path: 字典路径
    :return: 前缀词典（0代表前缀词，1代表真词）
    """
    prefix_dict = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            word = line.split()[0]
            for i in range(1, len(word)):
                if word[:i] not in prefix_dict:  # 不能用前缀覆盖词
                    prefix_dict[word[:i]] = 0  # 前缀
            prefix_dict[word] = 1  # 词
    return prefix_dict


def forward_max_match_method2(text, prefix_dict):
    """
    正向最大匹配 方法2，加载前缀词典
    :param text: 等待切词的文本
    :param prefix_dict: 前缀词典
    :return: 切词列表
    """
    words = []  # 准备用于放入切好的词
    if text == "":
        return []
    start_index, end_index = 0, 1  # 记录窗口的起始位置
    window = text[start_index:end_index]  # 从第一个字开始
    find_word = window  # 将第一个字先当做默认词
    while start_index < len(text):
        # 窗口没有在词典里出现
        if window not in prefix_dict or end_index > len(text):
            words.append(find_word)  # 记录找到的词
            start_index += len(find_word)  # 更新起点的位置
            end_index = start_index + 1
            window = text[start_index:end_index]  # 从新的位置开始一个字一个字向后找
            find_word = window
        # 窗口是一个词
        elif prefix_dict[window] == 1:
            find_word = window  # 查找到了一个词，还要在看有没有比他更长的词
            end_index += 1
            window = text[start_index:end_index]
        # 窗口是一个前缀
        elif prefix_dict[window] == 0:
            end_index += 1
            window = text[start_index:end_index]
    # 最后找到的window如果不在词典里，把单独的字加入切词结果
    if prefix_dict.get(window) != 1:
        words += list(window)
    else:
        words.append(window)
    return words


# ③最大反向匹配---窗口滑动
def backward_max_match(text, dict_path, max_len):
    """
    反向最大匹配 方法1，窗口滑动
    :param text: 输入字符串
    :param dict_path: 字典路径
    :param max_len: 滑动窗口的最大长度
    :return: 切词后的列表
    """
    words = []
    word_dict = {}
    with open(dict_path, encoding="utf8") as file:
        # 打开txt路径，构建字典
        for line in file:
            word = line.split()[0]
            word_dict[word] = 0
    while text != '':
        lens = min(max_len, len(text))
        word = text[-lens:]
        while word not in word_dict:
            if len(word) == 1:
                break
            word = word[-(len(word) - 1):]
        words.append(word)
        text = text[:-len(word)]
    words.reverse()
    return words


def main(text):
    result = []
    # ①最大正向匹配---窗口滑动
    temp1 = forward_max_match_method1(text=text, dict_path='forward_dict.txt', max_len=4)
    result.append(temp1)

    # ②最大正向匹配---前缀词典
    prefix_dict = load_prefix_word_dict('forward_dict.txt')
    temp2 = forward_max_match_method2(text=text, prefix_dict=prefix_dict)
    result.append(temp2)

    # ③最大反向匹配---窗口滑动
    temp3 = backward_max_match(text=text, dict_path='forward_dict.txt', max_len=4)
    result.append(temp3)

    # ④结巴分词
    temp4 = jieba.lcut(text)
    result.append(temp4)

    # ⑤结巴分词---加载自定义字典
    jieba.load_userdict('jieba_dict.txt')
    temp5 = jieba.lcut(text)
    result.append(temp5)

    # ⑥加载预训练的RNN神经网络模型进行切词
    temp6 = RNN_cut_word.predict("model.pth", "chars.txt", [text])[0]
    result.append(temp6)

    # ⑦字符串拆解
    temp7 = [i for i in text]
    result.append(temp7)

    return result


if __name__ == '__main__':
    sentence = "经常有意见分歧"
    print("原句子：\t",sentence)
    result = main(sentence)
    print("分词结果：")
    print(result)
    pass
