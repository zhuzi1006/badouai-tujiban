from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import json
import jieba
from gensim.models import Word2Vec
import numpy as np


LABELS = {'健康': 0, '军事': 1, '房产': 2, '社会': 3, '国际': 4, '旅游': 5, '彩票': 6, '时尚': 7, '文化': 8, '汽车': 9,
          '体育': 10, '家居': 11, '教育': 12, '娱乐': 13, '科技': 14, '股票': 15, '游戏': 16, '财经': 17}


def build_data(path, model):
    sentences = []
    labels = []
    with open(path, encoding='utf8') as f:
        for line in f:
            line = json.loads(line)
            label, title = line["tag"], line["title"]
            sentences.append(" ".join(jieba.lcut(title)))
            labels.append(label)
    data_X = sentence_to_vector(sentences, model)
    data_y = labels_to(labels)
    return data_X, data_y


def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


def sentence_to_vector(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)   # 如果word不在model中，则用0向量代替
        vectors.append(vector / len(words))

    return np.array(vectors)


def labels_to(labels):
    y = [LABELS[label] for label in labels]
    return y


def main():
    model = load_word2vec_model('model.w2v')
    train_X, train_y = build_data('train_tag_news.json', model)
    valid_X, valid_y = build_data('valid_tag_news.json', model)

    # std = StandardScaler()
    clf = GaussianNB()
    clf.fit(train_X, train_y)
    acc = clf.score(valid_X, valid_y)
    print('验证集上准确率 acc = %.3f' % acc)
    return model, clf


if __name__ == '__main__':
    model, clf = main()







