# 修改样本生成条件，实现三分类，字符串中包含2个相同字母为第0类；包含3个相同字母为第1类；其余为第2类
# 准确率大幅度下降至0.5左右，经查看是由于第1类样本的生成条件过于苛刻，导致生成样本数量少，模型从第1类样本学到的信息少
# 做了两处修改：1.调整nn.functional中的weight；2.缩减vocab
# 准确率提升至0.7左右
# 请教下老师怎么解决这个问题，进一步提升准确率？
import torch
import torch.nn as nn
import numpy as np
import random
import json


def build_vocab():
    # chars = "abcdefghijklmnopqrstuvwxyz"
    chars = "abcdefghi"
    vocab = {}
    for i, char in enumerate(chars, start=1):
        vocab[char] = i

    vocab['unk'] = len(vocab) + 1
    return vocab


# 字符串如果中有2个相同的字母，则为类0；如果有3个相同的字母，则为类1；其余为类2
def build_sample(sentence_len, vocab):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_len)]
    charN = {}
    for i, char in enumerate(set(x)):
        charN[char] = x.count(char)

    charNMAX = max(set(charN.values()))
    if charNMAX == 2:
        y = 0
    elif charNMAX == 3:
        y = 1
    else:
        y = 2
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y


def build_dataset(batch_size, sentence_len, vocab):
    feature = []
    target = []
    for i in range(batch_size):
        x, y = build_sample(sentence_len, vocab)
        feature.append(x)
        target.append([y])
    return torch.LongTensor(feature), torch.LongTensor(target)


class TorchModel(nn.Module):
    def __init__(self, vocab, input_dim, sentence_len):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab)+1, input_dim)
        self.layer = nn.Linear(input_dim, input_dim)
        self.pool = nn.MaxPool1d(sentence_len)
        self.classify = nn.Linear(input_dim, 3)
        self.activation = torch.sigmoid
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)
        x = self.layer(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.pool(x.transpose(1, 2)).squeeze()
        x = self.classify(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y.squeeze(), weight=torch.tensor([1.0, 1.5, 1.5]))
        else:
            return y_pred


def build_model(vocab, char_dim, sentence_len):
    model = TorchModel(vocab, char_dim, sentence_len)
    return model


def evaluate(model, vocab, sample_len):
    model.eval()
    total = 200
    x, y = build_dataset(total, sample_len, vocab)
    y = y.squeeze()
    print("A类样本数量：%d，B类样本数量：%d，C类样本数量：%d" % (y.tolist().count(0), y.tolist().count(1), y.tolist().count(2)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d / %d, 正确率： %f" % (correct, total, correct/(correct+wrong)))
    return correct / (correct + wrong)


def main():
    epoch_num = 15
    batch_size = 20
    train_sample = 2000
    char_dim = 50
    sentence_len = 6
    vocab = build_vocab()
    model = build_model(vocab, char_dim, sentence_len)
    optim = torch.optim.Adam(model.parameters(), lr=0.005)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, sentence_len, vocab)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("========\n第%d轮平均loss：%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_len)
        log.append([acc, np.mean(watch_loss)])
    torch.save(model.state_dict(), 'model.pth')
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

def predict(model_path, vocab_path, input_strings):
    char_dim = 50
    sentence_len = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_len)
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print(int(torch.argmax(result[i])), input_string, result[i])


if __name__ == "__main__":
    main()
    # test_strings = {"abihia", "abcdef", "defghi", "aaabbb"}
    # predict("model.pth", "vocab.json", test_strings)



