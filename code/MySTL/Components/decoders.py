# -*- coding: utf-8 -*-

"""
different kinds of extract_encoders
    the normal decoders has two modes: train and predict.
    the extract decoders only have one modes (like the train mode above).
"""

__author__ = 'PKU ChaiZ'

import tensorflow as tf


def _get_LSTMcell(num_units):
    lstm_cell = tf.contrib.rnn.LSTMCell(
                    num_units,
                    initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2)
                )
    return lstm_cell


def get_LSTM_decoder(word_emb, word_emb_size,
                     inp_num_seq, inp_seq_len, batch_size, max_batch_len,
                     init_state,
                     cell_size, layer_size,
                     inp_keep, oup_keep,
                     enc_out,
                     ext_att_info, ext_info_num, extra_dim,
                     mem_vec):
    """
        multi-layer LSTM decoder
    """

    # decoder
    with tf.variable_scope('LSTM_decoder'):
        # Cell
        cell_raw = tf.contrib.rnn.MultiRNNCell([_get_LSTMcell(cell_size) for _ in range(layer_size)])
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell_raw,
            input_keep_prob=inp_keep,
            output_keep_prob=oup_keep
        )

        # Input Layer
        #   shape = (max_batch_len, batch_size)
        inp_num_seq = tf.transpose(inp_num_seq, [1, 0])
        #   shape = (max_batch_len, batch_size, word_emb_size)
        inp_emb_seq = tf.nn.embedding_lookup(word_emb, inp_num_seq)
        inp_ta = tf.TensorArray(dtype=tf.float32, size=max_batch_len)
        inp_ta = inp_ta.unstack(inp_emb_seq)

        # used for extra attention
        W = tf.Variable(tf.random_normal([cell_size, extra_dim]))
        batch_W = tf.reshape(
            tf.tile(W, [batch_size, 1]),
            [batch_size, cell_size, extra_dim]
        )

        def loop_fn(time, cell_output, cell_state, loop_state):
            elements_finished = (time >= inp_seq_len)
            finished = tf.reduce_all(elements_finished)
            next_input = tf.cond(
                finished,
                lambda: tf.zeros([batch_size, word_emb_size], dtype=tf.float32),
                lambda: inp_ta.read(time))
            next_loop_state = None

            if cell_output is None:
                next_cell_state = init_state
                emit_output = cell_output
            else:
                next_cell_state = cell_state

                # 1. Ordinary Attention 
                cell_out_3dim = tf.reshape(cell_output, [-1, 1, cell_size])
                #   cell_out_3dim shape = (batch_size, 1, cell_size)
                #   enc_out.T     shape = (batch_size, cell_size, max_batch_len)
                #   energy_3dim   shape = (batch_size, 1, max_batch_len)
                energy_3dim = tf.matmul(cell_out_3dim, enc_out, adjoint_b=True)
                energy_2dim = tf.reshape(energy_3dim, [-1, max_batch_len])
                probs_2dim = tf.nn.softmax(energy_2dim)
                probs_3dim = tf.reshape(probs_2dim, [-1, 1, max_batch_len])
                #   probs_3dim   shape = (batch_size, 1, max_batch_len
                #   enc_out      shape = (batch_size, max_batch_len, cell_size)
                #   context_3dim shape = (batch_size, 1, cell_size)
                context_3dim = tf.matmul(probs_3dim, enc_out)
                #   shape = (batch_size, cell_size)
                ordinary_context = tf.reshape(context_3dim, [-1, cell_size])

                # 2. Extra Attention
                #   cell_out_3dim   shape = (batch_size, 1, cell_size)
                #   batch_W         shape = (batch_size, cell_size, extra_dim)
                #   ext_att_info.T  shape = (batch_size, extra_dim, extra_info_num)
                #   ext_energy_3dim shape = (batch_size, 1, extra_info_num)
                #
                ext_energy_3dim = tf.matmul(
                                      tf.matmul(cell_out_3dim, batch_W),
                                      ext_att_info,
                                      adjoint_b=True
                                  )
                ext_energy_2dim = tf.reshape(ext_energy_3dim, [-1, ext_info_num])
                ext_probs_2dim = tf.nn.softmax(ext_energy_2dim)
                ext_probs_3dim = tf.reshape(ext_probs_2dim, [-1, 1, ext_info_num])
                #   ext_probs_3dim shape = (batch_size, 1, ext_info_num)
                #   ext_att_info   shape = (batch_size, ext_info_num, extra_dim)
                #   context_3dim shape = (batch_size, 1, extra_dim)
                ext_context_3dim = tf.matmul(ext_probs_3dim, ext_att_info)
                #   shape = (batch_size, extra_dim)
                extra_context = tf.reshape(ext_context_3dim, [-1, extra_dim])

                # 3. Memory Information
                #   shape = (batch_size, mem_ndim)
                memory_context = mem_vec

                ############################
                # Mix and Get Final Output #
                ############################
                mix_output = tf.concat([cell_output,
                                        ordinary_context,
                                        extra_context,
                                        memory_context
                                        ], 1)
                emit_output = tf.contrib.layers.fully_connected(
                                  mix_output,
                                  cell_size,
                                  activation_fn=None
                              )

            return elements_finished, next_input, next_cell_state, emit_output, next_loop_state

        output_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
        output_time = output_ta.stack()
        # shape = (batch_size, max(inp_seq_len), cell_size)
        output_batch = tf.transpose(output_time, [1, 0, 2])

        return output_batch
