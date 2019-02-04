# -*- coding: utf-8 -*-

"""

"""

__author__ = 'PKU ChaiZ'


import jieba as jb
import numpy as np


def _get_comments(file_name):
    """
    read all the text data in a file

    :param file_name: the text file, each sentence in a line
    :return : a list, each element is a sentence
    """
    with open(file_name, 'r') as f:
        return [(item.strip()) for item in f]


def load_extra_data(file_name, word_2_int):
    """
    read the extra data file,
    extract all its words and add into the initial word_2_int
    turn each line of this file into a int_sequence

    :param file_name: the extra data file
    :param word_2_int: the initial dictionary
    :return: the extra data in int_sequence form
             the final dictionary
    """
    data_text = _get_comments(file_name)
    data_int_seq = []
    for item in data_text:
        word_seq = list(jb.cut(item, cut_all=False))
        int_seq = []
        for word in word_seq:
            if word not in word_2_int:
                word_2_int[word] = len(word_2_int)
            int_seq.append(word_2_int[word])
        data_int_seq.append(int_seq)

    return data_int_seq, word_2_int


def deal_extra_data(data_seq_init, threshold):
    """
    pick up the num_sequences whose length is shorter than threshold
    padding with <PAD>
    generate the length information
    """
    # NOTE: WE ASSUME <PAD> IS 0 HERE!
    # IF NOT, CHANGE THE CODE!
    data_seq = []
    # pick up
    for item in data_seq_init:
        if len(item) <= threshold:
            data_seq.append(item)
    # generate length information
    data_seq_len = [len(seq) for seq in data_seq]
    data_len_max = max(data_seq_len)
    # padding
    data_seq_padded = np.zeros(shape=[len(data_seq), data_len_max], dtype=np.int32)
    for index_i in range(len(data_seq)):
        for index_j in range(data_seq_len[index_i]):
            data_seq_padded[index_i][index_j] = data_seq[index_i][index_j]

    return data_seq_padded, data_seq_len


if __name__ == '__main__':
    data, words_2_int = load_extra_data(r'checking_data/e_data', {'Word0': 0, 'Word1': 1, 'Word2': 2})
    print(data)
    print(words_2_int)

    data_padded, data_len = deal_extra_data(data, 2)
    print(data_padded)
    print(data_len)
