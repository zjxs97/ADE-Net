# -*- coding: utf-8 -*-

"""
different kinds of loss functions
"""

__author__ = 'PKU ChaiZ'

import tensorflow as tf


def get_inp_tensors(word_emb_dim, mem_size):
    # input and targets
    #   shape = (batch_size, max_inp_seq_len)
    inp_num_seq = tf.placeholder(shape=(None, None), dtype=tf.int32, name='inp_num_seq')
    #   shape = (batch_size,)
    inp_seq_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='inp_seq_len')
    #   shape = (batch_size, max_inp_seq_len)
    tgt_num_seq = tf.placeholder(shape=(None, None), dtype=tf.int32, name='tgt_num_seq')

    # word embeddings
    word_emb = tf.placeholder(shape=(None, word_emb_dim), dtype=tf.float32, name='word_emb')

    # dropouts
    enc_inp_keep = tf.placeholder(shape=(), dtype=tf.float32, name='enc_trn_keep')
    enc_oup_keep = tf.placeholder(shape=(), dtype=tf.float32, name='enc_trn_keep')
    dec_inp_keep = tf.placeholder(shape=(), dtype=tf.float32, name='enc_trn_keep')
    dec_oup_keep = tf.placeholder(shape=(), dtype=tf.float32, name='enc_trn_keep')

    # used for attention
    ext_att_seq = tf.placeholder(shape=(None, None), dtype=tf.int32, name='ext_att_ifo')
    ext_att_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='ext_att_len')

    # used for memory
    support_seqs = tf.placeholder(shape=(None, mem_size, None), dtype=tf.int32, name='ext_att_ifo')

    # used for classification
    classify_tgt = tf.placeholder(shape=(None,), dtype=tf.float32, name='classify_tgt')
    classify_drop0 = tf.placeholder(shape=(), dtype=tf.float32, name='classify_drop0')
    classify_drop1 = tf.placeholder(shape=(), dtype=tf.float32, name='classify_drop1')

    # used for exponential learning rate decay
    learning_rate = tf.placeholder(shape=(), dtype=tf.float32, name='learning_rate')
    current_epoch = tf.placeholder(shape=(), dtype=tf.float32, name='current_epoch')
    decay_rate = tf.placeholder(shape=(), dtype=tf.float32, name='decay_rate')

    return inp_num_seq, inp_seq_len, tgt_num_seq, word_emb, \
        enc_inp_keep, enc_oup_keep, dec_inp_keep, dec_oup_keep, \
        ext_att_seq, ext_att_len, support_seqs, \
        classify_tgt, classify_drop0, classify_drop1, \
        learning_rate, current_epoch, decay_rate


def get_att_input(word_emb,
                  ext_att_seq, ext_att_len,
                  extra_dim,
                  batch_size):
    #   shape = (extra_info_num, extra_len_threshold, word_emb_size)
    ext_att_emb = tf.nn.embedding_lookup(word_emb, ext_att_seq)
    #   shape = (extra_info_num, word_emb_size)
    ext_att_sum = tf.reduce_sum(ext_att_emb, axis=1)
    #   shape = (extra_info_num, 1)
    ext_divider = tf.cast(tf.reshape(ext_att_len, [-1, 1]), tf.float32)
    ext_att_avg = ext_att_sum / ext_divider
    #   shape = (extra_info_num, extra_dim)
    ext_att_dse = tf.contrib.layers.fully_connected(ext_att_avg, extra_dim)
    #   shape = (extra_info_num * batch_size, extra_dim)
    ext_att_tle = tf.tile(ext_att_dse, [batch_size, 1])
    #   shape = (batch_size, extra_info_num, extra_dim)
    ext_att_msg = tf.reshape(ext_att_tle, [batch_size, -1, extra_dim])

    return ext_att_msg



