# -*- coding: utf-8 -*-

"""
different kinds of encoders
"""

__author__ = 'PKU ChaiZ'

import tensorflow as tf


def _get_LSTMcell(num_units):
    lstm_cell = tf.contrib.rnn.LSTMCell(
                    num_units,
                    initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2)
                )
    return lstm_cell


def get_LSTM_encoder(word_emb, inp_num_seq, inp_seq_len,
                     cell_size, layer_size,
                     inp_keep, oup_keep):
    """
        multi-layer LSTM encoder
    """

    with tf.variable_scope('LSTM_encoder'):
        # Cell
        cell_raw = tf.contrib.rnn.MultiRNNCell([_get_LSTMcell(cell_size) for _ in range(layer_size)])
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell_raw,
            input_keep_prob=inp_keep,
            output_keep_prob=oup_keep
        )
        # Each Layer
        inp_emb_seq = tf.nn.embedding_lookup(word_emb, inp_num_seq)
        output, final_state = tf.nn.dynamic_rnn(
                                  cell,
                                  inp_emb_seq,
                                  sequence_length=inp_seq_len,
                                  dtype=tf.float32,
                                  time_major=False
                              )
    # Returns
    return output, final_state


def get_BiLSTM_encoder(word_emb, inp_num_seq, inp_seq_len,
                       cell_size, layer_size,
                       inp_keep, oup_keep):
    """
        multi-layer BiLSTM encoder
    """

    with tf.variable_scope('BiLSTM_encoder'):
        # Cell
        cell_raw = tf.contrib.rnn.MultiRNNCell([_get_LSTMcell(cell_size) for _ in range(layer_size)])
        cell = tf.nn.rnn_cell.DropoutWrapper(
                   cell_raw,
                   input_keep_prob=inp_keep,
                   output_keep_prob=oup_keep
               )
        # Each Layer
        inp_emb_seq = tf.nn.embedding_lookup(word_emb, inp_num_seq)
        (fw_output, bw_output), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
                                                           cell, cell, inp_emb_seq,
                                                           sequence_length=inp_seq_len,
                                                           dtype=tf.float32,
                                                           time_major=False
                                                       )
        # Output
        #   shape = (batch_size, encoder_max_time, 2 * enc_cell_size)
        output = tf.concat((fw_output, bw_output), 2)
        # Final_state
        final_state_list = [tf.contrib.rnn.LSTMStateTuple(
                                c=tf.concat((fw_state[layer_num].c, bw_state[layer_num].c), 1),
                                h=tf.concat((fw_state[layer_num].h, bw_state[layer_num].h), 1)
                            )
                            for layer_num in range(layer_size)]
        final_state = tuple(final_state_list)
    # Returns
    return output, final_state
