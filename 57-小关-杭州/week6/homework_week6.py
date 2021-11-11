import torch
import math
import os
from nnlm import build_model, build_vocab
from collections import defaultdict
import math
from collections import defaultdict
import numpy as np



def load_trained_language_model(path='./model/财经.pth'):
    char_dim = 128
    window_size = 6
    vocab = build_vocab("vocab.txt")
    model = build_model(vocab, char_dim)
    model.load_state_dict(torch.load(path))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    model.window_size = window_size
    model.vocab = vocab

    return model

# 建立同音字表
def build_confusion(path='tongyin.txt'):
    confusion = {}
    with open(path, encoding='utf-8') as f:
        for index, line in enumerate(f):
            chars = line[:-1]
            chars = chars.split(" ")
            confusion_char = []
            for char in chars[1]:
                confusion_char.append(char)
            confusion[chars[0]] = confusion_char    #{字: 同音字}

    return confusion


def sentence_prob(sentence, model):
    prob = 0
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - model.window_size)
            window = sentence[start: i]
            x = [model.vocab.get(char, model.vocab["<UNK>"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = model.vocab.get(target, model.vocab["<UNK>"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0]
            target_prob = pred_prob_distribute[target_index]
            prob += -target_prob * math.log(target_prob, 10)    # 成句概率用熵来表示

    return prob


def confusion_prob(sentence, confusion, model):
    prob = sentence_prob(sentence, model)
    errata = defaultdict(list)
    for index, char in enumerate(sentence):
        if char not in list(confusion.keys()):
            continue
        else:
            for idx, new_char in enumerate(list(confusion[char])):
                new_sentence = sentence.replace(sentence[index], new_char)
                new_prob = sentence_prob(new_sentence, model)
                diff = new_prob - prob
                errata[char].append(np.array(diff))

    return errata


def main():
    sentence = "在圈球货币体系出现危季帝情况下"
    model = load_trained_language_model()
    confusion = build_confusion()
    errata = confusion_prob(sentence, confusion, model)
    result = []
    for char, diff in errata.items():
        min_diff = min(diff)
        min_diff_idx = diff.index(min_diff)
        result.append([char, min_diff, min_diff_idx])
    result = sorted(result, key=lambda x: x[1]) # 将熵减幅度最大的字置前
    print('-' * 30)
    print(result[0][0], "->", confusion[result[0][0]][result[0][2]])
    print(result[1][0], "->", confusion[result[1][0]][result[1][2]])
    print('-' * 30)


if __name__ == "__main__":
    main()

