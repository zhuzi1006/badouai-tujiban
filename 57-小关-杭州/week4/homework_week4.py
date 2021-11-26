import jieba
from gensim.models import Word2Vec
import numpy as np
import os.path
import math
from sklearn.cluster import KMeans
from collections import defaultdict

def load_corpus(path):
    corpus = []
    with open(path, encoding='utf8') as f:
        for line in f:
            corpus.append(jieba.lcut(line))
    return corpus

def train_word2vec_model(corpus, dim=100):
    model = Word2Vec(corpus, vector_size=dim)
    model.save('model.w2v')
    return model

def load_sentences(path):
    sentences = set()
    with open(path, encoding='utf8') as f:
        for line in f:
            sentences.add(" ".join(jieba.cut(line.strip())))
    return sentences

def sentences_to_vector(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    # 检查当前目录下词向量模型是否存在
    if os.path.isfile('model.w2v'):
        model = Word2Vec.load('model.w2v')
    else:
        corpus = load_corpus('corpus.txt')
        model = train_word2vec_model(corpus)

    sentences = load_sentences('titles.txt')
    vectors = sentences_to_vector(sentences, model)

    n_clusters = int(math.sqrt(len(sentences)))
    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)

    sentences_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        sentences_label_dict[label].append(sentence)
    for label, sentences in sentences_label_dict.items():
        print('cluster id is %d' % label)
        for i in range(min(4, len(sentences))):
            print(sentences[i].replace(" ", ""))
        print("----------")


if __name__ == "__main__":
    main()
