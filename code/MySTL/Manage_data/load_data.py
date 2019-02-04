# -*- coding: utf-8 -*-

"""
Some functions to load and convert data
"""

__author__ = 'PKU ChaiZ'


import re
import jieba as jb
import numpy as np
import math


#####################
# Part 1: load_data #
#####################
def _get_comments(file_name):
    """
    read all the text data in a file

    :param file_name: the text file, each sentence in a line
    :return : a list, each element is a sentence
    """
    with open(file_name, 'r') as f:
        return [(item.strip()) for item in f]


def _convert_data_based_on_letter(labeled_comments, letter_2_int):
    """
    turn a labeled-text list into (x = num_seq, y = bin_seq),
        (1) in x and y, each element is an int
                for x, it corresponds with a letter
                for y, it corresponds with 'b','i','n'
        (2) the corresponding relationship is shown in dictionary
                for x, it is letter_2_int
                for y, 0 for 'n', 1 for 'b', 2 for 'i'

    :param labeled_comments: a list of labeled_text_comments
           letter_2_int_x: the original dictionary
    :return: the num_seq, bin_seq and letter_2_int_x
    """
    whole_num_seq = []
    whole_bin_seq = []

    for labeled_comment in labeled_comments:
        # generate the number sequence
        comment_num_seq = []
        #   pure text comment
        str_del = re.compile('[\[\]]')
        raw_comment = str_del.sub('', labeled_comment[:])
        #   make it into num sequence
        for letter in raw_comment:
            if letter not in letter_2_int:
                letter_2_int[letter] = len(letter_2_int)
            comment_num_seq.append(letter_2_int[letter])
        whole_num_seq.append(comment_num_seq)

        # generate the bin sequence
        #   generate an "i", "n" str
        comment_in_str = ''
        flag_in_bracket = 0
        for letter in labeled_comment:
            if letter == '[':
                flag_in_bracket = 1
            elif letter == ']':
                flag_in_bracket = 0
            else:
                if flag_in_bracket:
                    comment_in_str += 'i'
                else:
                    comment_in_str += 'n'
        #   generate an "b", "i", "n" str
        compile_ni = re.compile('ni')
        comment_bin_str = compile_ni.sub('nb', comment_in_str)
        if comment_bin_str[0] == 'i':
            comment_bin_str = 'b' + comment_bin_str[1:]
        #   generate the bin sequence
        comment_bin_seq = []
        for letter in comment_bin_str:
            if letter == 'n':
                comment_bin_seq.append(0)
            elif letter == 'b':
                comment_bin_seq.append(1)
            elif letter == 'i':
                comment_bin_seq.append(2)
        whole_bin_seq.append(comment_bin_seq)

    assert(len(whole_num_seq) == len(whole_bin_seq))
    return whole_num_seq, whole_bin_seq, letter_2_int


def _convert_data_based_on_word(labeled_comments, word_2_int):
    """
    similar to _convert_data_based_on_letter,
    but each element in num_seq corresponds with a word
    """
    # we need the letter_bin_seq to help generating word_bin_seq
    _, whole_letter_bin_seq, _ = _convert_data_based_on_letter(labeled_comments, {})
    assert(len(whole_letter_bin_seq) == len(labeled_comments))

    whole_num_seq = []
    whole_word_bin_seq = []

    for comment_index in range(len(labeled_comments)):
        labeled_comment = labeled_comments[comment_index]

        # generate the number sequence
        comment_num_seq = []
        #   pure text comment
        str_del = re.compile('[\[\]]')
        raw_comment = str_del.sub('', labeled_comment[:])
        #   cut into words
        word_text_seq = list(jb.cut(raw_comment, cut_all=False))
        for word in word_text_seq:
            if word not in word_2_int:
                word_2_int[word] = len(word_2_int)
            comment_num_seq.append(word_2_int[word])
        whole_num_seq.append(comment_num_seq)

        # generate the bin sequence
        comment_letter_bin_seq = whole_letter_bin_seq[comment_index]
        comment_word_bin_seq = []
        #   generate an "i", "n" str
        comment_in_str = ''
        start = 0
        for word in word_text_seq:
            word_bin = []
            # if there is 'b' or 'i' in corresponding word_bin,
            # this word must be 'b' or 'i'
            for _index in range(len(word)):
                word_bin.append(comment_letter_bin_seq[start])
                start += 1
            if (1 in word_bin) or (2 in word_bin):
                comment_in_str += 'i'
            else:
                comment_in_str += 'n'
        #   generate an "b", "i", "n" str
        compile_ni = re.compile('ni')
        comment_bin_str = compile_ni.sub('nb', comment_in_str)
        if comment_bin_str[0] == 'i':
            comment_bin_str = 'b' + comment_bin_str[1:]
        #   generate the bin sequence
        for letter in comment_bin_str:
            if letter == 'n':
                comment_word_bin_seq.append(0)
            elif letter == 'b':
                comment_word_bin_seq.append(1)
            elif letter == 'i':
                comment_word_bin_seq.append(2)
        whole_word_bin_seq.append(comment_word_bin_seq)

    assert(len(whole_num_seq) == len(whole_word_bin_seq))
    return whole_num_seq, whole_word_bin_seq, word_2_int


def _k_cross_split_generator(data_set, k=10):
    """
    split a data-set into k-cross to perform k-cross validation
    """

    total_size = len(data_set)
    block_size = int(total_size / k)
    shuffle_index = np.arange(total_size)
    np.random.shuffle(shuffle_index)
    for turn in range(k):
        start, end = turn * block_size, min(total_size, (turn + 1) * block_size)
        test_set = [data_set[shuffle_index[i]] for i in range(start, end)]
        train_set = [data_set[shuffle_index[i]] for i in range(total_size) if not(start <= i < end)]
        yield train_set, test_set


def get_cross_valid_generator(p_file_name, n_file_name, init_element_2_int, base="word", k=10):
    """
    turn the labeled_text into (num_seq, bin_seq) based on word/letter with k_cross
    the element in num_seq and text's relation is in element_2_int
    """

    # read comments
    p_comments = _get_comments(p_file_name)
    n_comments = _get_comments(n_file_name)

    # numerate comments and generate bin sequences
    if base == "letter":
        p_x, p_y, element_2_int_half = _convert_data_based_on_letter(p_comments, init_element_2_int)
        n_x, n_y, element_2_int = _convert_data_based_on_letter(n_comments, element_2_int_half)
        int_2_element = dict(zip(element_2_int.values(), element_2_int.keys()))
    else:
        p_x, p_y, element_2_int_half = _convert_data_based_on_word(p_comments, init_element_2_int)
        n_x, n_y, element_2_int = _convert_data_based_on_word(n_comments, element_2_int_half)
        int_2_element = dict(zip(element_2_int.values(), element_2_int.keys()))

    # use comments pair to get the generator
    p_pairs = list(zip(p_x, p_y))
    n_pairs = list(zip(n_x, n_y))

    k_cross_p_generator = _k_cross_split_generator(p_pairs, k)
    k_cross_n_generator = _k_cross_split_generator(n_pairs, k)

    for k_iter in range(k):
        p_train_pairs, p_test_pairs = next(k_cross_p_generator)
        n_train_pairs, n_test_pairs = next(k_cross_n_generator)
        un_shuffled_train_pairs = p_train_pairs + n_train_pairs
        un_shuffled_test_pairs = p_test_pairs + n_test_pairs


        p_train_len = len(p_train_pairs)
        p_test_len = len(p_test_pairs)
        n_train_len = len(n_train_pairs)
        n_test_len = len(n_test_pairs)
        train_len = len(un_shuffled_train_pairs)
        test_len = len(un_shuffled_test_pairs)

        train_shuffle_index = np.arange(train_len)
        np.random.shuffle(train_shuffle_index)
        test_shuffle_index = np.arange(test_len)
        np.random.shuffle(test_shuffle_index)
        train_pairs = [un_shuffled_train_pairs[train_shuffle_index[i]] for i in range(train_len)]
        test_pairs = [un_shuffled_test_pairs[test_shuffle_index[i]] for i in range(test_len)]

        # all train data in this cross
        train_x = [train_pairs[index][0] for index in range(train_len)]
        train_ye = [train_pairs[index][1] for index in range(train_len)]
        train_yc = [int(1 in train_ye[index]) for index in range(train_len)]

        # positive train data in this cross
        p_train_x = [p_train_pairs[index][0] for index in range(p_train_len)]
        p_train_ye = [p_train_pairs[index][1] for index in range(p_train_len)]
        p_train_yc = [int(1 in p_train_ye[index]) for index in range(p_train_len)]

        # negative train data in this cross
        n_train_x = [n_train_pairs[index][0] for index in range(n_train_len)]
        n_train_ye = [n_train_pairs[index][1] for index in range(n_train_len)]
        n_train_yc = [int(1 in n_train_ye[index]) for index in range(n_train_len)]

        # all test data in this cross
        test_x = [test_pairs[index][0] for index in range(test_len)]
        test_ye = [test_pairs[index][1] for index in range(test_len)]
        test_yc = [int(1 in test_ye[index]) for index in range(test_len)]

        # positive train data in this cross
        p_test_x = [p_test_pairs[index][0] for index in range(p_test_len)]
        p_test_ye = [p_test_pairs[index][1] for index in range(p_test_len)]
        p_test_yc = [int(1 in p_test_ye[index]) for index in range(p_test_len)]

        # negative train data in this cross
        n_test_x = [n_test_pairs[index][0] for index in range(n_test_len)]
        n_test_ye = [n_test_pairs[index][1] for index in range(n_test_len)]
        n_test_yc = [int(1 in n_test_ye[index]) for index in range(n_test_len)]

        # returns
        yield int_2_element, \
            train_x, train_yc, train_ye, test_x, test_yc, test_ye, \
            p_train_x, p_train_yc, p_train_ye, p_test_x, p_test_yc, p_test_ye, \
            n_train_x, n_train_yc, n_train_ye, n_test_x, n_test_ye, n_test_ye


#####################
# Part 2: get_batch #
#####################
def find_support_seqs(num_seq, support_size,
                      int_2_element,
                      comment_2_medicine, medicine_2_comments):
    # For robust
    support_data_robust = []
    for i in range(support_size):
        support_data_robust.append(['1'])

    # First, recover the text comment
    comment = ''
    for num in num_seq:
        comment += int_2_element[num]

    # Next, find the comment's medicine
    if comment not in comment_2_medicine:
        print('Warning: {} not in comment_2_medicine'.format(comment))
        return support_data_robust, len(num_seq)
    medname = comment_2_medicine[comment]

    # Then, find all the same medicine's comments
    if medname not in medicine_2_comments:
        print('Warning: {} not in medicine_2_comments'.format(medname))
        return support_data_robust, len(num_seq)
    related_comments = medicine_2_comments[medname]

    # Turn the related comments to num_seq
    element_2_int = dict(zip(int_2_element.values(), int_2_element.keys()))
    support_data = []
    for text in related_comments:
        this_num_seq = []
        cut_text = list(jb.cut(text, cut_all=False))
        for element in cut_text:
            this_num_seq.append(element_2_int[element])
        support_data.append(this_num_seq)
    support_len = [len(item) for item in support_data]
    if len(support_data) >= support_size:
        support_data = support_data[:support_size]
        support_len = support_len[:support_size]
    else:
        while len(support_data) < support_size:
            support_data.append(support_data[0])
    max_len = max(support_len)

    return support_data, max_len


def get_batch_generator(data_x, data_yc, data_ye,
                        batch_size, max_batch_length, memory_size,
                        int_2_element,
                        comment_2_medicine, medicine_2_comments):
    """
    load the (x, y) into batch_size blocks and padding in each batch
    """
    total_size = len(data_x)
    batch_num = int(math.ceil(total_size / batch_size))

    shuffle_index = np.arange(total_size)
    np.random.shuffle(shuffle_index)

    for turn in range(batch_num):
        #######################
        # Part-1 Get Raw Data #
        #######################
        #   raw data
        start, end = turn * batch_size, min((turn + 1) * batch_size, total_size)
        raw_x = [data_x[shuffle_index[index]] for index in range(start, end)]
        raw_yc = [data_yc[shuffle_index[index]] for index in range(start, end)]
        raw_ye = [data_ye[shuffle_index[index]] for index in range(start, end)]
        #   raw length information
        batch_len = len(raw_x)
        batch_elements_len = [len(seq) + 1 for seq in raw_x]
        max_data_len = max(batch_elements_len)

        ###########################################
        # Part-2 Get Support data (for MemoryNet) #
        ###########################################
        max_support_len = 0
        raw_support = []
        for data in raw_x:
            support_seqs, max_len = find_support_seqs(data, memory_size,
                                                      int_2_element,
                                                      comment_2_medicine,
                                                      medicine_2_comments)
            raw_support.append(support_seqs)
            max_support_len = max(max_support_len, max_len)

        ###################
        # Part-3 Pad Data #
        ###################
        assert max(max_data_len, max_support_len) <= max_batch_length, \
            "the Constant max_batch_length is too small!"

        #  1. pad the length information
        #       if is not enough for a batch, simply copy the first data
        while len(batch_elements_len) < batch_size:
            batch_elements_len.append(batch_elements_len[0])

        #  2. pad the data (WE ASSUME THE <PAD> = 0, <EOS> = 1)
        #    (1) add <PAD> and <EOS>
        x = np.zeros(shape=[batch_size, max_batch_length], dtype=np.int32)
        ye = np.zeros(shape=[batch_size, max_batch_length], dtype=np.int32)
        yc = np.zeros(shape=[batch_size], dtype=np.int32)
        for index_i in range(batch_len):
            yc[index_i] = raw_yc[index_i]
            j_max = len(raw_x[index_i])
            for index_j in range(j_max):
                x[index_i, index_j] = raw_x[index_i][index_j]
                ye[index_i, index_j] = raw_ye[index_i][index_j]
            x[index_i, j_max] = 1
        #    (2) if is not enough for a batch, simply copy the first data
        for index_i in range(batch_len, batch_size):
            x[index_i] = x[0]
            ye[index_i] = ye[0]
            yc[index_i] = yc[0]

        # 3. pad the support information
        #    (1) add <PAD>
        support = np.zeros(shape=[batch_size, memory_size, max_batch_length], dtype=np.int32)
        for index_batch in range(batch_len):
            for index_memory in range(memory_size):
                j_max = len(raw_support[index_batch][index_memory])
                for index_j in range(j_max):
                    support[index_batch, index_memory, index_j] = raw_support[index_batch][index_memory][index_j]
        #    (2) if is not enough for a batch, simply copy the first data
        for index_batch in range(batch_len, batch_size):
            support[index_batch] = support[0]

        yield(x, yc, ye, batch_elements_len, support)


######################
# Part 3: split_data #
######################
def split_data(data, rate):
    """
    split a list into two part, rate : 1
    when generate k-cross data, we have already shuffled
    so here we don't shuffle anymore
    """
    total_size = len(data)
    split_point = int(total_size * rate / (rate + 1))
    part1 = [data[index] for index in range(0, split_point)]
    part2 = [data[index] for index in range(split_point, total_size)]

    return part1, part2


if __name__ == '__main__':
    print("test for part 1")
    data_gen = get_cross_valid_generator(r'checking_data/p_data', r'checking_data/n_data', {'PAD': 0, 'EOS': 1})
    i2e, \
    trn_x, trn_yc, trn_ye, tst_x, tst_yc, tst_ye, \
    p_trn_x, p_trn_yc, p_trn_ye, p_tst_x, p_tst_yc, p_tst_ye, \
    n_trn_x, n_trn_yc, n_trn_ye, n_tst_x, n_tst_yc, n_tst_ye, \
        = next(data_gen)
    print(len(trn_x), len(trn_yc), len(trn_ye), len(tst_x), len(tst_yc), len(tst_ye))
    print(len(p_trn_x), len(p_trn_yc), len(p_trn_ye), len(p_tst_x), len(p_tst_yc), len(p_tst_ye))
    print(len(n_trn_x), len(n_trn_yc), len(n_trn_ye), len(n_tst_x), len(n_tst_yc), len(n_tst_ye))
    print('\n')
    print(i2e)
    print(trn_x)
    print(trn_yc)
    print(trn_ye)
    print(tst_x)
    print(tst_yc)
    print(tst_ye)
    print('\n')
    print(p_trn_x)
    print(p_trn_yc)
    print(p_trn_ye)
    print(p_tst_x)
    print(p_tst_yc)
    print(p_tst_ye)
    print('\n')
    print(n_trn_x)
    print(n_trn_yc)
    print(n_trn_ye)
    print(n_tst_x)
    print(n_tst_yc)
    print(n_tst_ye)
    print('\n')

    print("test for part 3")
    part_1_x, part_2_x = split_data(tst_x, 1)
    print(part_1_x)
    print(part_2_x)
