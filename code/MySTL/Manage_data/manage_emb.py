# -*- coding: utf-8 -*-

"""
  use a message-file (produced by the Google_word2vect from some support corpus)
to get word-embedding matrix from a word-ID dictionary
"""

__author__ = 'PKU ChaiZ'

import numpy as np


def _get_prepared_emb(file_name, dim):
    """
        read a resulting-file produced by the Google_word2vect,
        return the words and vectors in the form of a dictionary.
    """
    whole_words_dict = {}
    with open(file_name, 'r') as f:
        first_line = ((f.readline()).strip()).split()
        assert len(first_line) == 2, 'first line is wrong!'
        # word_num = int(first_line[0])
        word_dim = int(first_line[1])
        assert word_dim == dim, 'constant dim is wrong!'
        while True:
            line = f.readline()
            if not line:
                break
            splited_line = (line.strip()).split()
            word_part = splited_line[0]
            vect_part = splited_line[1:]
            if word_dim == len(vect_part):
                whole_words_dict[word_part] = vect_part
    return whole_words_dict


def get_emb(file_name, dim, int_2_element):
    """
        given an dictionary, which maps the word to its ID
        using a message-file (produced by the Google_word2vect)
        word_emb: the final word_embedding matrix
            if the word is in the message-file,
                word_embedding[word's ID] comes from the message_file
            else:
                word_embedding[word's ID] is random
        num: how many words in the dictionary is in the message-file
            (how many word embeddings we can get from the message-file)
    """
    prepared_emb = _get_prepared_emb(file_name, dim)
    element_2_int = dict(zip(int_2_element.values(), int_2_element.keys()))
    word_emb = 0.2 * np.random.random((len(int_2_element), dim)) - 0.1
    num = 0
    for word in prepared_emb:
        if word in element_2_int:
            num += 1
            index = element_2_int[word]
            number = prepared_emb[word]
            for i in range(dim):
                word_emb[index][i] = number[i]
    return word_emb, num


if __name__ == '__main__':
    emb, num = get_emb(
                 r'checking_data/emb_data', 300,
                 {0: 'PAD', 1: 'EOS', 2: '跟着', 3: '没有', 4: '三罐', 5: '多天'}
               )
    print(num)
    print(emb)
