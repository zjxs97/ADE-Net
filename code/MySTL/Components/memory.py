# -*- coding: utf-8 -*-

"""
different kinds of loss functions
"""

__author__ = 'PKU ChaiZ'

import tensorflow as tf


def _conv_pool(input, W, b, filter_depth, filter_size, seq_len):
    # (batch_size, seq_shape, 1, filter_depth)
    conv = tf.nn.relu(
        tf.nn.bias_add(
            tf.nn.conv2d(
                input, W, strides=[1, 1, 1, 1], padding='VALID'
            ),
            b
        )
    )
    # (batch_size, 1, 1, filter_depth)
    pool = tf.nn.max_pool(
        conv, ksize=[1, seq_len - (filter_size - 1), 1, 1], strides=[1, 1, 1, 1], padding='VALID'
    )
    # (batch_size, filter_depth)
    result = tf.reshape(pool, [-1, filter_depth])
    return result


def get_each_embedding(query_seq, mem_seqs,
                       seq_len, word_emb_len,
                       mem_size, edim):
    ############################################
    # CNN parameters, filter_depth from 1 to 6 #
    ############################################
    assert edim % 3 == 0, "edim of mem can't divided by 3"
    filter_depth = int(edim / 3)

    a_w1 = tf.get_variable('a_w1', [1, word_emb_len, 1, filter_depth], initializer=tf.contrib.layers.xavier_initializer_conv2d())
    a_w2 = tf.get_variable('a_w2', [2, word_emb_len, 1, filter_depth], initializer=tf.contrib.layers.xavier_initializer_conv2d())
    a_w3 = tf.get_variable('a_w3', [3, word_emb_len, 1, filter_depth], initializer=tf.contrib.layers.xavier_initializer_conv2d())
    
    b_w1 = tf.get_variable('b_w1', [1, word_emb_len, 1, filter_depth], initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b_w2 = tf.get_variable('b_w2', [2, word_emb_len, 1, filter_depth], initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b_w3 = tf.get_variable('b_w3', [3, word_emb_len, 1, filter_depth], initializer=tf.contrib.layers.xavier_initializer_conv2d())
    
    c_w1 = tf.get_variable('c_w1', [1, word_emb_len, 1, filter_depth], initializer=tf.contrib.layers.xavier_initializer_conv2d())
    c_w2 = tf.get_variable('c_w2', [2, word_emb_len, 1, filter_depth], initializer=tf.contrib.layers.xavier_initializer_conv2d())
    c_w3 = tf.get_variable('c_w3', [3, word_emb_len, 1, filter_depth], initializer=tf.contrib.layers.xavier_initializer_conv2d())
    
    a_b1 = tf.get_variable('a_b1', [filter_depth], initializer=tf.constant_initializer(0.01))
    a_b2 = tf.get_variable('a_b2', [filter_depth], initializer=tf.constant_initializer(0.01))
    a_b3 = tf.get_variable('a_b3', [filter_depth], initializer=tf.constant_initializer(0.01))
    
    b_b1 = tf.get_variable('b_b1', [filter_depth], initializer=tf.constant_initializer(0.01))
    b_b2 = tf.get_variable('b_b2', [filter_depth], initializer=tf.constant_initializer(0.01))
    b_b3 = tf.get_variable('b_b3', [filter_depth], initializer=tf.constant_initializer(0.01))
    
    c_b1 = tf.get_variable('c_b1', [filter_depth], initializer=tf.constant_initializer(0.01))
    c_b2 = tf.get_variable('c_b2', [filter_depth], initializer=tf.constant_initializer(0.01))
    c_b3 = tf.get_variable('c_b3', [filter_depth], initializer=tf.constant_initializer(0.01))
    
    ###################################################################
    # Encode query sequence                                           #
    #  from (batch_size, seq_len, word_emb_len) to (batch_size, edim) #
    ###################################################################
    # (batch_size, seq_len, word_emb_len, 1)
    query_seq_4_CNN = tf.reshape(query_seq, [-1, seq_len, word_emb_len, 1])
    # each c: (batch_size, filter_depth)
    c1 = _conv_pool(query_seq_4_CNN, c_w1, c_b1, filter_depth, 1, seq_len)
    c2 = _conv_pool(query_seq_4_CNN, c_w2, c_b2, filter_depth, 2, seq_len)
    c3 = _conv_pool(query_seq_4_CNN, c_w3, c_b3, filter_depth, 3, seq_len)
    # emb_C:  (batch_size, edim)
    emb_C = tf.concat([c1, c2, c3], -1)
    
    #########################################################
    # Encode memory into A and B                            #
    #   from: (batch_size, mem_size, seq_len, word_emb_len) #
    #   to:   (batch_size, men_size, edim)                  #
    #########################################################
    # (batch_size * mem_size, seq_len, word_emb_len, 1)
    mem_seqs_4_CNN = tf.reshape(mem_seqs, [-1, seq_len, word_emb_len, 1])

    a1 = _conv_pool(mem_seqs_4_CNN, a_w1, a_b1, filter_depth, 1, seq_len)
    a2 = _conv_pool(mem_seqs_4_CNN, a_w2, a_b2, filter_depth, 2, seq_len)
    a3 = _conv_pool(mem_seqs_4_CNN, a_w3, a_b3, filter_depth, 3, seq_len)
    # emb_A:  (batah_size, mem_size, edim)
    emb_A = tf.reshape(
                tf.concat([a1, a2, a3], -1),
                [-1, mem_size, edim]
            )

    mem_seqs_4_CNN = tf.reshape(mem_seqs, [-1, seq_len, word_emb_len, 1])
    b1 = _conv_pool(mem_seqs_4_CNN, b_w1, b_b1, filter_depth, 1, seq_len)
    b2 = _conv_pool(mem_seqs_4_CNN, b_w2, b_b2, filter_depth, 2, seq_len)
    b3 = _conv_pool(mem_seqs_4_CNN, b_w3, b_b3, filter_depth, 3, seq_len)
    # emb_B:  (batah_size, mem_size, edim)
    emb_B = tf.reshape(
        tf.concat([b1, b2, b3], -1),
        [-1, mem_size, edim]
    )

    return emb_A, emb_B, emb_C


def core_multi_layer(nhop, mem_size, edim,
                     Ain, Bin, Cin,
                    ):
    hid = [Cin]
    for h in range(nhop):
        # 1. Input Layer, the embedded query, (batch_size, edim)
        hid3dim = tf.reshape(hid[-1], [-1, 1, edim])
        # 2. Attention Layer
        #   get energies
        Aout = tf.matmul(hid3dim, Ain, adjoint_b=True)
        Aout2dim = tf.reshape(Aout, [-1, mem_size])
        #   get the probability by softmax the energies
        P = tf.nn.softmax(Aout2dim)
        probs3dim = tf.reshape(P, [-1, 1, mem_size])
        #   get the context-vector by adding using the probability
        Bout = tf.matmul(probs3dim, Bin)
        Bout2dim = tf.reshape(Bout, [-1, edim])
        # 3. Residual Layer, (batch_size, edim)
        Res = tf.add(hid[-1], Bout2dim)
        hid.append(Res)

    return hid[-1]


def memory_network(word_emb, query_seq, support_seqs,
                   mem_size, seq_len, word_emb_len,
                   nhop, edim):
    # shape = (batch_size, seq_max_time, word_emb_size)
    query_seq_emb = tf.nn.embedding_lookup(word_emb, query_seq)
    # shape = (batch_size, mem_size, seq_max_time, word_emb_size)
    mem_seqs_emb = tf.nn.embedding_lookup(word_emb, support_seqs)

    emb_A, emb_B, emb_C = get_each_embedding(
                              query_seq_emb, mem_seqs_emb,
                              seq_len, word_emb_len,
                              mem_size, edim
                          )
    # shape = (batch_size, edim)
    embedded_out = core_multi_layer(
                       nhop, mem_size, edim,
                       emb_A, emb_B, emb_C
                   )

    return embedded_out
