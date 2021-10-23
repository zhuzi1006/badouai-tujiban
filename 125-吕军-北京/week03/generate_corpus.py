# This is a sample Python script.
import random

"""
根据词表随机生成文本
"""

def load_word_dict(path):
    words = set()
    chars = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            word = line.split()[0]
            words.add(word)
            for char in word:
                chars.add(char)
    return words, chars

def generate_corpus(col, row, words, chars, path):
    words, chars = list(words), list(chars)
    writer = open(path, "w", encoding="utf8")
    for col_index in range(col):
        sentence = ""
        for row_index in range(row):
            if random.random() > 0.9:
                sentence += random.choice(chars)
            else:
                sentence += random.choice(words)
        writer.write(sentence + "\n")
    writer.close()
    return

if __name__ == "__main__":
    words, chars = load_word_dict("dict.txt")
    # print(words, chars)
    generate_corpus(100000, 10, words, chars, "test.txt")