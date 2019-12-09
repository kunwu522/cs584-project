import os
import numpy as np


def load_glove_weights(filepath):
    glove_weights = {}
    with open(filepath, 'r') as fp:
        for line in fp:
            line = line.split(' ')
            word = line[0].strip().lower()
            vector = np.array([float(f.strip()) for f in line[1:]])
            glove_weights[word.lower()] = vector
    return glove_weights


def generate_weights(glove_weights, word2idx, embedding_size=300):
    weights = np.zeros((len(word2idx) + 1, embedding_size))
    for word, idx in word2idx.items():
        if word in glove_weights:
            weights[word2idx[word]] = np.array(glove_weights[word])
    
    return weights
