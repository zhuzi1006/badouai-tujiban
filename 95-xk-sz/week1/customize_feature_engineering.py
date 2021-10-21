import json
import os
import random
import torch
import numpy as np



def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {}
    for index, char in enumerate(chars):
        idx = index + 1
        vocab[char] = str(idx)  # 每个字对应一个序号
    vocab['unk'] = str(len(vocab) + 1)
    return vocab


# 建立数据集
# 输入需要的样本数量。需要多少生成多少
def build_dataset(raw, raw_id, vocab, sample_length, sentence_length):
    with open(raw, 'w') as wf, open(raw_id, 'w') as wf_id:
        for _ in range(sample_length):
            x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
            if "q" in x:
                y = '1'
            else:
                y = '0'
            ids = [vocab.get(word, vocab['unk']) for word in x]
            raw_line, raw_id_line = ' '.join([y] + x), ' '.join([y] + ids)
            wf.write(raw_line)
            wf.write('\n')
            wf_id.write(raw_id_line)
            wf_id.write('\n')


def gen_raw_id_feature_map(raw, raw_id, feature_map_file, sample_length=10000, sentence_length=6):
    feature_map = build_vocab()
    with open(feature_map_file, "w", encoding="utf8") as wf:
        wf.write(json.dumps(feature_map, ensure_ascii=False, indent=2))
    build_dataset(raw, raw_id, feature_map, sample_length, sentence_length)


if __name__ == '__main__':
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    gen_raw_id_feature_map('raw.txt', 'raw_id.txt', 'feature_map.txt')
    xy = np.loadtxt('raw_id.txt', delimiter=' ', dtype=np.long)
    x_data = torch.from_numpy(xy[:, 1:])
    y_data = torch.from_numpy(xy[:, [0]])  # 加中括号是为了保持维度
    print(x_data.size(), y_data.size())
    deal_dataset = TensorDataset(x_data, y_data)
    train_loader = DataLoader(deal_dataset, batch_size=100, shuffle=True)

