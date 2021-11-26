import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset


def get_vocab(path):
    rf = open(path)
    vocab = json.load(rf)
    return vocab


def get_data_set(id_data_path, sample_len, shuffle=True):
    xy = np.loadtxt(id_data_path, delimiter=' ')
    x_data = torch.from_numpy(xy[:, 1:]).long()
    y_data = torch.from_numpy(xy[:, [0]]).float()  # 加中括号是为了保持维度
    deal_dataset = TensorDataset(x_data, y_data)
    return DataLoader(deal_dataset, batch_size=sample_len, shuffle=shuffle)
