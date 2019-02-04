# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import sys
import MySTL.Manage_data.load_data as mld
import MySTL.Manage_data.judge_result as mjr
import MySTL.Manage_data.manage_extra as mme
import MySTL.Manage_data.manage_emb as mmeb
import MySTL.Manage_data.manage_memory as mmm
import MySTL.Components.encoders as my_Encs
import MySTL.Components.decoders as my_Decs
import MySTL.Components.memory as my_Mem
import MySTL.Components.classify as my_Classify
import MySTL.Components.helpers as my_Helper

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
log_dir = r'./Result/'

# Hyper Parameters
train_data_type = 'PN'
test_data_type = 'PN'

classify_loss_rate = 1
extract_loss_rate = 1

learning_rate_pre = 0.005
decay_rate_pre = 0.9
learning_rate_trn = 0.003
decay_rate_trn = 0.95
decay_step = 1

batch_size = 32

pre_epoch = 0
epoch = 30
early_stopping = True

enc_cell_size = 50
dec_cell_size = 100
enc_layer_size = 2
dec_layer_size = 2

enc_input_keep = 0.9
enc_output_keep = 0.9
dec_input_keep = 0.5
dec_output_keep = 0.5
classify_dropout_0 = 0.6
classify_dropout_1 = 0.6

extra_len_threshold = 5
extra_dim = 100

memory_size = 50
mem_nhop = 2
mem_edim = 60

# Constants
max_batch_length = 100
word_emb_size = 300
max_gradient_norm = 5
cross_K = 10
special_symbols = {'PAD': 0, 'EOS': 1}
class_num = 3
bin_base = "word"
p_file_name = r'./Data/p_data'
n_file_name = r'./Data/n_data'
extra_file_name = r'./Data/e_data'
emb_file_name = r'./Data/emb_data'
com2med_file_name = r'./Data/mem_data'

if __name__ == '__main__':
    final_dir = sys.argv[1]
    print('final_dir: {}'.format(final_dir))
    f_final = open(final_dir, 'w')

    ###########################################################################
    #                 Part 1 - Build the Computing Graph                      #
    ###########################################################################

    ##################
    # Get Input Info #
    ##################
    inp_num_seq, inp_seq_len, tgt_num_seq, word_emb, \
        enc_inp_keep, enc_oup_keep, dec_inp_keep, dec_oup_keep, \
        ext_att_seq, ext_att_len, support_seqs, \
        classify_tgt, classify_drop0, classify_drop1, \
        learning_rate, current_epoch, decay_rate \
        = my_Helper.get_inp_tensors(word_emb_size, memory_size)

    ####################
    # (Shared) Encoder #
    ####################
    enc_output, enc_state = my_Encs.get_BiLSTM_encoder(
                                word_emb, inp_num_seq, inp_seq_len,
                                enc_cell_size, enc_layer_size,
                                enc_inp_keep, enc_oup_keep
                            )
    
    ############
    # Classify #
    ############
    # 1. Get model's output
    y = my_Classify.get_classify_layers(
            enc_state[-1].h,
            classify_drop0, classify_drop1
        )
    # 2. Get loss function
    y_ = tf.reshape(classify_tgt, [-1, 1])
    loss_classify = -tf.reduce_mean(
                        y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) +
                        y * tf.log(tf.clip_by_value(y_, 1e-10, 1.0))
                    ) / tf.cast(batch_size, tf.float32)

    ###########
    # Extract #
    ###########
    # 1. Get Extra Information from Standard-Names
    #   shape = (batch_size, ext_info_num, extra_dim)
    ext_att_info = my_Helper.get_att_input(
                       word_emb,
                       ext_att_seq, ext_att_len,
                       extra_dim,
                       batch_size
                   )
    _, ext_info_num, _ = tf.unstack(tf.shape(ext_att_info))

    # 2. Get Context Vector from Memory-Network
    #   shape = (batch_size, mem_ndim)
    mem_context_vector = my_Mem.memory_network(
                             word_emb, inp_num_seq, support_seqs,
                             memory_size, max_batch_length, word_emb_size,
                             mem_nhop, mem_edim
                         )

    # 3. Get decoder outputs
    dec_output = my_Decs.get_LSTM_decoder(
                         word_emb, word_emb_size,
                         inp_num_seq, inp_seq_len, batch_size, max_batch_length,
                         enc_state,
                         dec_cell_size, dec_layer_size,
                         dec_inp_keep, dec_oup_keep,
                         enc_output,
                         ext_att_info, ext_info_num, extra_dim,
                         mem_context_vector
                     )

    # 4. Get the final logits, prediction and targets
    #   (batch_size, max(inp_seq_len), class_num)
    train_logits = tf.contrib.layers.fully_connected(
                       dec_output,
                       class_num,
                       activation_fn=None
                   )
    #   (batch_size, max(inp_seq_len))
    prediction = tf.argmax(train_logits, axis=2)

    # original target: shape = (batch_size, max_batch_length)
    # managet target:  shape = (batch_size, max(inp_seq_len))
    # why we need this convert?
    #   When we pad data, we directly use max_batch_length (this can simplify
    # the memory part), so actually we pad more zeros.
    #   However, the decoder terminates using the inp_seq_len information in
    # training_helper, so the output of decoder is max(inp_seq_len)
    #   So, we need to perform a conversion here.
    _, max_seq_len, _ = tf.unstack(tf.shape(train_logits))
    targets = tf.slice(tgt_num_seq, [0, 0], [batch_size, max_seq_len])

    # 5. Get loss Function
    masks = tf.sequence_mask(inp_seq_len, max_seq_len, dtype=tf.float32)
    loss_extract = tf.contrib.seq2seq.sequence_loss(
                       train_logits,
                       targets,
                       masks
                   )

    ############
    # Train_op #
    ############
    # Pre_Train op
    pre_loss = loss_extract
    #    Gradient Clipping
    pre_params = tf.trainable_variables()
    pre_gradients = tf.gradients(pre_loss, pre_params)
    pre_clipped_gradients, _ = tf.clip_by_global_norm(pre_gradients, max_gradient_norm)
    #   Optimizer
    pre_exp_decay_learning_rate = tf.train.exponential_decay(
        learning_rate, current_epoch, decay_step, decay_rate
    )
    pre_optimizer = tf.train.AdamOptimizer(pre_exp_decay_learning_rate)
    pre_train_op = pre_optimizer.apply_gradients(zip(pre_clipped_gradients, pre_params))

    # Formal Train op
    #   Optimization
    loss = classify_loss_rate * loss_classify + extract_loss_rate * loss_extract
    #   Gradient Clipping
    params = tf.trainable_variables()
    gradients = tf.gradients(loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
    #   Optimizer
    exp_decay_learning_rate = tf.train.exponential_decay(
        learning_rate, current_epoch, decay_step, decay_rate
    )
    optimizer = tf.train.AdamOptimizer(exp_decay_learning_rate)
    train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

    ###########################################################################
    #                   Part 2 - Train the Computing Graph                    #
    ###########################################################################
    # load extra information
    extra_info, extra_dict = mme.load_extra_data(extra_file_name, special_symbols)
    extra_attention_seq, extra_attention_len = mme.deal_extra_data(extra_info, extra_len_threshold)

    # perform cross_validation
    cross_num = 0
    cross_valid_generator = mld.get_cross_valid_generator(
                                p_file_name, n_file_name,
                                extra_dict, base=bin_base, k=cross_K
                            )

    epoch_log_tot_classify = []
    epoch_log_tot_strict = []
    epoch_log_tot_soft = []
    for epoch_index in range(epoch + 5):
        epoch_log_tot_classify.append(np.array([0, 0, 0], dtype=np.float32))
        epoch_log_tot_strict.append(np.array([0, 0, 0], dtype=np.float32))
        epoch_log_tot_soft.append(np.array([0, 0, 0], dtype=np.float32))
    early_stop_tot_classify = np.array([0, 0, 0], dtype=np.float32)
    early_stop_tot_strict = np.array([0, 0, 0], dtype=np.float32)
    early_stop_tot_soft = np.array([0, 0, 0], dtype=np.float32)

    for k_iter in range(cross_K):
        int_2_element, \
        all_trndev_x, all_trndev_yc, all_trndev_ye, all_tst_x, all_tst_yc, all_tst_ye, \
            p_trndev_x, p_trndev_yc, p_trndev_ye, p_tst_x, p_tst_yc, p_tst_ye, \
            n_trndev_x, n_trndev_yc, n_trndev_ye, n_tst_x, n_tst_yc, n_tst_ye = next(cross_valid_generator)

        assert (train_data_type == 'P') or (train_data_type == 'PN'), 'train_data_type wrong!'
        assert (test_data_type == 'P') or (test_data_type == 'PN'), 'train_data_type wrong!'
        # Notice: the split_data don't change the order, it's very important
        if train_data_type == 'P':
            trn_x, dev_x = mld.split_data(p_trndev_x, 8)
            trn_yc, dev_yc = mld.split_data(p_trndev_yc, 8)
            trn_ye, dev_ye = mld.split_data(p_trndev_ye, 8)
        else:
            trn_x, dev_x = mld.split_data(all_trndev_x, 8)
            trn_yc, dev_yc = mld.split_data(all_trndev_yc, 8)
            trn_ye, dev_ye = mld.split_data(all_trndev_ye, 8)

        if test_data_type == 'P':
            tst_x = p_tst_x
            tst_yc = p_tst_yc
            tst_ye = p_tst_ye
        else:
            tst_x = all_tst_x
            tst_yc = all_tst_yc
            tst_ye = all_tst_ye

        word_embedding, finded_num = mmeb.get_emb(emb_file_name, word_emb_size, int_2_element)
        com2med = mmm.get_comment_2_medicine(com2med_file_name)
        med2com = mmm.get_medicine_2_comment(com2med)

        # log file
        flog = open(log_dir + r'log{}.txt'.format(cross_num + 1), 'w')

        with tf.Session(config=config) as sess:
            ##############
            # Initialize #
            ##############
            print('cross {}, have {} / {} words embeddings'.format(cross_num + 1, finded_num, len(word_embedding)))
            sess.run(tf.global_variables_initializer())
            tf.set_random_seed(47)

            #############
            # Pre-Train #
            #############
            for pre_epoch_num in range(pre_epoch):
                print('cross{}-pre_epoch{}, '.format(cross_num + 1, pre_epoch_num + 1))
                flog.write('pre_epoch{},\n'.format(pre_epoch_num + 1))

                # pre_train our model using mini-batch
                batch_generator = mld.get_batch_generator(trn_x, trn_yc, trn_ye,
                                                          batch_size, max_batch_length, memory_size,
                                                          int_2_element, com2med, med2com)
                for batched_inp_num_seq, batched_tgt_classify, batched_tgt_num_seq, batched_inp_seq_len, batched_support in batch_generator:
                    fd_batched_train = {
                        inp_num_seq: batched_inp_num_seq,
                        tgt_num_seq: batched_tgt_num_seq,
                        inp_seq_len: batched_inp_seq_len,
                        word_emb: word_embedding,
                        enc_inp_keep: enc_input_keep,
                        enc_oup_keep: enc_output_keep,
                        dec_inp_keep: dec_input_keep,
                        dec_oup_keep: dec_output_keep,
                        ext_att_seq: extra_attention_seq,
                        ext_att_len: extra_attention_len,
                        support_seqs: batched_support,
                        learning_rate: learning_rate_pre,
                        decay_rate: decay_rate_pre,
                        current_epoch: pre_epoch_num
                    }
                    _, batched_train_loss = sess.run([pre_train_op, pre_loss], fd_batched_train)

                # show the total pre_train loss, strict/soft_F1
                trn_generator = mld.get_batch_generator(trn_x, trn_yc, trn_ye,
                                                        batch_size, max_batch_length, memory_size,
                                                        int_2_element, com2med, med2com)
                trn_strict_result = np.array([0, 0, 0], dtype=np.float32)
                trn_soft_result = np.array([0, 0, 0], dtype=np.float32)
                trn_extract_loss = 0.0
                trn_batch_num = 0
                for inp_num_seq_trn, tgt_classify_trn, tgt_num_seq_trn, inp_seq_len_trn, support_trn in trn_generator:
                    trn_batch_num += 1
                    fd_trn = {
                        inp_num_seq: inp_num_seq_trn,
                        tgt_num_seq: tgt_num_seq_trn,
                        inp_seq_len: inp_seq_len_trn,
                        word_emb: word_embedding,
                        enc_inp_keep: enc_input_keep,
                        enc_oup_keep: enc_output_keep,
                        dec_inp_keep: dec_input_keep,
                        dec_oup_keep: dec_output_keep,
                        ext_att_seq: extra_attention_seq,
                        ext_att_len: extra_attention_len,
                        support_seqs: support_trn
                    }
                    # Extract
                    trn_losse = sess.run(pre_loss, fd_trn)
                    trn_pred = sess.run(prediction, fd_trn)
                    strict_trn, soft_trn = mjr.judge_result(tgt_num_seq_trn, trn_pred)
                    trn_strict_result += np.array(strict_trn)
                    trn_soft_result += np.array(soft_trn)
                    trn_extract_loss += trn_losse

                trn_strict_result /= trn_batch_num
                trn_soft_result /= trn_batch_num
                trn_extract_loss /= trn_batch_num

                print('  pre_train loss: {:<10.6}'.format(trn_extract_loss))
                print('            st_F1: {:<8.4}, {:<8.4}, {:<8.4}'.
                           format(trn_strict_result[0], trn_strict_result[1], trn_strict_result[2]))
                print('            so_F1: {:<8.4}, {:<8.4}, {:<8.4}'.
                           format(trn_soft_result[0], trn_soft_result[1], trn_soft_result[2]))
                flog.write('  pre_train loss: {:<10.6}\n'.format(trn_extract_loss))
                flog.write('            st_F1: {:<8.4}, {:<8.4}, {:<8.4}\n'.
                           format(trn_strict_result[0], trn_strict_result[1], trn_strict_result[2]))
                flog.write('            so_F1: {:<8.4}, {:<8.4}, {:<8.4}\n'.
                           format(trn_soft_result[0], trn_soft_result[1], trn_soft_result[2]))

                # show the total pre_dev loss, strict/soft_F1
                dev_generator = mld.get_batch_generator(dev_x, dev_yc, dev_ye,
                                                        batch_size, max_batch_length, memory_size,
                                                        int_2_element, com2med, med2com)
                dev_strict_result = np.array([0, 0, 0], dtype=np.float32)
                dev_soft_result = np.array([0, 0, 0], dtype=np.float32)
                dev_extract_loss = 0.0
                dev_batch_num = 0
                for inp_num_seq_dev, tgt_classify_dev, tgt_num_seq_dev, inp_seq_len_dev, support_dev in dev_generator:
                    dev_batch_num += 1
                    fd_dev = {
                        inp_num_seq: inp_num_seq_dev,
                        tgt_num_seq: tgt_num_seq_dev,
                        inp_seq_len: inp_seq_len_dev,
                        word_emb: word_embedding,
                        enc_inp_keep: 1.0,
                        enc_oup_keep: 1.0,
                        dec_inp_keep: 1.0,
                        dec_oup_keep: 1.0,
                        ext_att_seq: extra_attention_seq,
                        ext_att_len: extra_attention_len,
                        support_seqs: support_dev
                    }
                    # Extract
                    dev_losse = sess.run(pre_loss, fd_dev)
                    dev_pred = sess.run(prediction, fd_dev)
                    strict_dev, soft_dev = mjr.judge_result(tgt_num_seq_dev, dev_pred)
                    dev_strict_result += np.array(strict_dev)
                    dev_soft_result += np.array(soft_dev)
                    dev_extract_loss += dev_losse

                dev_strict_result /= dev_batch_num
                dev_soft_result /= dev_batch_num
                dev_extract_loss /= dev_batch_num

                print('  pre_dev   loss: {:<10.6}'.format(dev_extract_loss))
                print('            st_F1: {:<8.4}, {:<8.4}, {:<8.4}'.
                           format(dev_strict_result[0], dev_strict_result[1], dev_strict_result[2]))
                print('            so_F1: {:<8.4}, {:<8.4}, {:<8.4}'.
                           format(dev_soft_result[0], dev_soft_result[1], dev_soft_result[2]))
                flog.write('  pre_dev   loss: {:<10.6}\n'.format(dev_extract_loss))
                flog.write('            st_F1: {:<8.4}, {:<8.4}, {:<8.4}\n'.
                           format(dev_strict_result[0], dev_strict_result[1], dev_strict_result[2]))
                flog.write('            so_F1: {:<8.4}, {:<8.4}, {:<8.4}\n'.
                           format(dev_soft_result[0], dev_soft_result[1], dev_soft_result[2]))

                # show the total pre_test loss, strict/soft_F1
                tst_generator = mld.get_batch_generator(tst_x, tst_yc, tst_ye,
                                                        batch_size, max_batch_length, memory_size,
                                                        int_2_element, com2med, med2com)
                tst_strict_result = np.array([0, 0, 0], dtype=np.float32)
                tst_soft_result = np.array([0, 0, 0], dtype=np.float32)
                tst_extract_loss = 0.0
                tst_batch_num = 0
                for inp_num_seq_tst, tgt_classify_tst, tgt_num_seq_tst, inp_seq_len_tst, support_tst in tst_generator:
                    tst_batch_num += 1
                    fd_tst = {
                        inp_num_seq: inp_num_seq_tst,
                        tgt_num_seq: tgt_num_seq_tst,
                        inp_seq_len: inp_seq_len_tst,
                        word_emb: word_embedding,
                        enc_inp_keep: 1.0,
                        enc_oup_keep: 1.0,
                        dec_inp_keep: 1.0,
                        dec_oup_keep: 1.0,
                        ext_att_seq: extra_attention_seq,
                        ext_att_len: extra_attention_len,
                        support_seqs: support_tst
                    }
                    # Extract
                    tst_losse = sess.run(pre_loss, fd_tst)
                    tst_pred = sess.run(prediction, fd_tst)
                    strict_tst, soft_tst = mjr.judge_result(tgt_num_seq_tst, tst_pred)
                    tst_strict_result += np.array(strict_tst)
                    tst_soft_result += np.array(soft_tst)
                    tst_extract_loss += tst_losse

                tst_strict_result /= tst_batch_num
                tst_soft_result /= tst_batch_num
                tst_extract_loss /= tst_batch_num

                print('  pre_test  loss: {:<10.6}'.format(tst_extract_loss))
                print('            st_F1: {:<8.4}, {:<8.4}, {:<8.4}'.
                           format(tst_strict_result[0], tst_strict_result[1], tst_strict_result[2]))
                print('            so_F1: {:<8.4}, {:<8.4}, {:<8.4}'.
                           format(tst_soft_result[0], tst_soft_result[1], tst_soft_result[2]))
                flog.write('  pre_test  loss: {:<10.6}\n'.format(tst_extract_loss))
                flog.write('            st_F1: {:<8.4}, {:<8.4}, {:<8.4}\n'.
                           format(tst_strict_result[0], tst_strict_result[1], tst_strict_result[2]))
                flog.write('            so_F1: {:<8.4}, {:<8.4}, {:<8.4}\n'.
                           format(tst_soft_result[0], tst_soft_result[1], tst_soft_result[2]))
                flog.flush()

            #########
            # Train #
            #########
            # used for early stopping
            # recording the dev loss of last time
            early_stop_classify_result = np.array([0, 0, 0], dtype=np.float32)
            early_stop_strict_result = np.array([0, 0, 0], dtype=np.float32)
            early_stop_soft_result = np.array([0, 0, 0], dtype=np.float32)
            old_dev_soft_result = np.array([0, 0, 0], dtype=np.float32)
            old_dev_strict_result = np.array([0, 0, 0], dtype=np.float32)
            lowest_loss = 0.0
            lowest_count = 0

            for epoch_num in range(epoch):
                print('cross{}-epoch{}, '.format(cross_num + 1, epoch_num + 1))
                flog.write('epoch{}\n'.format(epoch_num + 1))

                # train our model using mini-batch
                batch_generator = mld.get_batch_generator(trn_x, trn_yc, trn_ye,
                                                          batch_size, max_batch_length, memory_size,
                                                          int_2_element, com2med, med2com)
                for batched_inp_num_seq, batched_tgt_classify, batched_tgt_num_seq, batched_inp_seq_len, batched_support in batch_generator:
                    fd_batched_train = {
                        inp_num_seq: batched_inp_num_seq,
                        tgt_num_seq: batched_tgt_num_seq,
                        inp_seq_len: batched_inp_seq_len,
                        word_emb: word_embedding,
                        enc_inp_keep: enc_input_keep,
                        enc_oup_keep: enc_output_keep,
                        dec_inp_keep: dec_input_keep,
                        dec_oup_keep: dec_output_keep,
                        ext_att_seq: extra_attention_seq,
                        ext_att_len: extra_attention_len,
                        support_seqs: batched_support,
                        classify_tgt: batched_tgt_classify,
                        classify_drop0: classify_dropout_0,
                        classify_drop1: classify_dropout_1,
                        learning_rate: learning_rate_trn,
                        decay_rate: decay_rate_trn,
                        current_epoch: epoch_num
                    }
                    _, batched_train_loss = sess.run([train_op, loss], fd_batched_train)

                # show the total train loss, strict/soft_F1
                trn_generator = mld.get_batch_generator(trn_x, trn_yc, trn_ye,
                                                        batch_size, max_batch_length, memory_size,
                                                        int_2_element, com2med, med2com)
                trn_total_loss = 0.0
                trn_strict_result = np.array([0, 0, 0], dtype=np.float32)
                trn_soft_result = np.array([0, 0, 0], dtype=np.float32)
                trn_extract_loss = 0.0
                trn_classify_result = np.array([0, 0, 0], dtype=np.float32)
                trn_classify_loss = 0.0
                trn_batch_num = 0
                for inp_num_seq_trn, tgt_classify_trn, tgt_num_seq_trn, inp_seq_len_trn, support_trn in trn_generator:
                    trn_batch_num += 1
                    fd_trn = {
                        inp_num_seq: inp_num_seq_trn,
                        tgt_num_seq: tgt_num_seq_trn,
                        inp_seq_len: inp_seq_len_trn,
                        word_emb: word_embedding,
                        enc_inp_keep: enc_input_keep,
                        enc_oup_keep: enc_output_keep,
                        dec_inp_keep: dec_input_keep,
                        dec_oup_keep: dec_output_keep,
                        ext_att_seq: extra_attention_seq,
                        ext_att_len: extra_attention_len,
                        support_seqs: support_trn,
                        classify_tgt: tgt_classify_trn,
                        classify_drop0: classify_dropout_0,
                        classify_drop1: classify_dropout_1
                    }
                    # Total loss
                    trn_loss = sess.run(loss, fd_trn)
                    trn_total_loss += trn_loss

                    # Classify
                    trn_lossc = sess.run(loss_classify, fd_trn)
                    # standard
                    trn_y_ = sess.run(y_, fd_trn)
                    trn_y_ = np.reshape(trn_y_, [-1])
                    # prediction
                    trn_y = sess.run(y, fd_trn)
                    trn_y = np.reshape(trn_y, [-1])
                    trn_rslt = [int(item > 0.5) for item in trn_y]
                    # train classify information
                    trn_result = mjr.judge_classify(trn_y_, trn_rslt)
                    trn_classify_result += np.array(trn_result)
                    trn_classify_loss += trn_lossc

                    # Extract
                    trn_losse = sess.run(loss_extract, fd_trn)
                    trn_pred = sess.run(prediction, fd_trn)
                    strict_trn, soft_trn = mjr.judge_result(tgt_num_seq_trn, trn_pred)
                    trn_strict_result += np.array(strict_trn)
                    trn_soft_result += np.array(soft_trn)
                    trn_extract_loss += trn_losse

                trn_classify_result /= trn_batch_num
                trn_strict_result /= trn_batch_num
                trn_soft_result /= trn_batch_num
                trn_classify_loss /= trn_batch_num
                trn_extract_loss /= trn_batch_num
                trn_total_loss /= trn_batch_num

                print('  train loss: {:<10.6}, ({:<10.6}, {:<10.6})'.
                           format(trn_total_loss, trn_classify_loss, trn_extract_loss))
                print('           c: {:<8.4}, {:<8.4}, {:<8.4}'.
                           format(trn_classify_result[0], trn_classify_result[1], trn_classify_result[2]))
                print('        e_st: {:<8.4}, {:<8.4}, {:<8.4}'.
                           format(trn_strict_result[0], trn_strict_result[1], trn_strict_result[2]))
                print('        e_so: {:<8.4}, {:<8.4}, {:<8.4}'.
                           format(trn_soft_result[0], trn_soft_result[1], trn_soft_result[2]))

                flog.write('  train loss: {:<10.6}, ({:<10.6}, {:<10.6})\n'.
                           format(trn_total_loss, trn_classify_loss, trn_extract_loss))
                flog.write('           c: {:<8.4}, {:<8.4}, {:<8.4}\n'.
                           format(trn_classify_result[0], trn_classify_result[1], trn_classify_result[2]))
                flog.write('        e_st: {:<8.4}, {:<8.4}, {:<8.4}\n'.
                           format(trn_strict_result[0], trn_strict_result[1], trn_strict_result[2]))
                flog.write('        e_so: {:<8.4}, {:<8.4}, {:<8.4}\n'.
                           format(trn_soft_result[0], trn_soft_result[1], trn_soft_result[2]))

                # show the total develop loss, strict/soft_F1
                dev_generator = mld.get_batch_generator(dev_x, dev_yc, dev_ye,
                                                        batch_size, max_batch_length, memory_size,
                                                        int_2_element, com2med, med2com)
                dev_total_loss = 0.0
                dev_strict_result = np.array([0, 0, 0], dtype=np.float32)
                dev_soft_result = np.array([0, 0, 0], dtype=np.float32)
                dev_extract_loss = 0.0
                dev_classify_result = np.array([0, 0, 0], dtype=np.float32)
                dev_classify_loss = 0.0
                dev_batch_num = 0
                for inp_num_seq_dev, tgt_classify_dev, tgt_num_seq_dev, inp_seq_len_dev, support_dev in dev_generator:
                    dev_batch_num += 1
                    fd_dev = {
                        inp_num_seq: inp_num_seq_dev,
                        tgt_num_seq: tgt_num_seq_dev,
                        inp_seq_len: inp_seq_len_dev,
                        word_emb: word_embedding,
                        enc_inp_keep: 1.0,
                        enc_oup_keep: 1.0,
                        dec_inp_keep: 1.0,
                        dec_oup_keep: 1.0,
                        ext_att_seq: extra_attention_seq,
                        ext_att_len: extra_attention_len,
                        support_seqs: support_dev,
                        classify_tgt: tgt_classify_dev,
                        classify_drop0: 1.0,
                        classify_drop1: 1.0
                    }
                    # Total loss
                    dev_loss = sess.run(loss, fd_dev)
                    dev_total_loss += dev_loss

                    # Classify
                    dev_lossc = sess.run(loss_classify, fd_dev)
                    # standard
                    dev_y_ = sess.run(y_, fd_dev)
                    dev_y_ = np.reshape(dev_y_, [-1])
                    # prediction
                    dev_y = sess.run(y, fd_dev)
                    dev_y = np.reshape(dev_y, [-1])
                    dev_rslt = [int(item > 0.5) for item in dev_y]
                    # train classify information
                    dev_result = mjr.judge_classify(dev_y_, dev_rslt)
                    dev_classify_result += np.array(dev_result)
                    dev_classify_loss += dev_lossc

                    # Extract
                    dev_losse = sess.run(loss_extract, fd_dev)
                    dev_pred = sess.run(prediction, fd_dev)
                    strict_dev, soft_dev = mjr.judge_result(tgt_num_seq_dev, dev_pred)
                    dev_strict_result += np.array(strict_dev)
                    dev_soft_result += np.array(soft_dev)
                    dev_extract_loss += dev_losse

                dev_classify_result /= dev_batch_num
                dev_strict_result /= dev_batch_num
                dev_soft_result /= dev_batch_num
                dev_classify_loss /= dev_batch_num
                dev_extract_loss /= dev_batch_num
                dev_total_loss /= dev_batch_num

                print('  dev   loss: {:<10.6}, ({:<10.6}, {:<10.6})'.
                           format(dev_total_loss, dev_classify_loss, dev_extract_loss))
                print('           c: {:<8.4}, {:<8.4}, {:<8.4}'.
                           format(dev_classify_result[0], dev_classify_result[1], dev_classify_result[2]))
                print('        e_st: {:<8.4}, {:<8.4}, {:<8.4}'.
                           format(dev_strict_result[0], dev_strict_result[1], dev_strict_result[2]))
                print('        e_so: {:<8.4}, {:<8.4}, {:<8.4}'.
                           format(dev_soft_result[0], dev_soft_result[1], dev_soft_result[2]))

                flog.write('  dev   loss: {:<10.6}, ({:<10.6}, {:<10.6})\n'.
                           format(dev_total_loss, dev_classify_loss, dev_extract_loss))
                flog.write('           c: {:<8.4}, {:<8.4}, {:<8.4}\n'.
                           format(dev_classify_result[0], dev_classify_result[1], dev_classify_result[2]))
                flog.write('        e_st: {:<8.4}, {:<8.4}, {:<8.4}\n'.
                           format(dev_strict_result[0], dev_strict_result[1], dev_strict_result[2]))
                flog.write('        e_so: {:<8.4}, {:<8.4}, {:<8.4}\n'.
                           format(dev_soft_result[0], dev_soft_result[1], dev_soft_result[2]))

                # show the total train loss, strict/soft_F1
                tst_generator = mld.get_batch_generator(tst_x, tst_yc, tst_ye,
                                                        batch_size, max_batch_length, memory_size,
                                                        int_2_element, com2med, med2com)
                tst_total_loss = 0.0
                tst_strict_result = np.array([0, 0, 0], dtype=np.float32)
                tst_soft_result = np.array([0, 0, 0], dtype=np.float32)
                tst_extract_loss = 0.0
                tst_classify_result = np.array([0, 0, 0], dtype=np.float32)
                tst_classify_loss = 0.0
                tst_batch_num = 0
                for inp_num_seq_tst, tgt_classify_tst, tgt_num_seq_tst, inp_seq_len_tst, support_tst in tst_generator:
                    tst_batch_num += 1
                    fd_tst = {
                        inp_num_seq: inp_num_seq_tst,
                        tgt_num_seq: tgt_num_seq_tst,
                        inp_seq_len: inp_seq_len_tst,
                        word_emb: word_embedding,
                        enc_inp_keep: 1.0,
                        enc_oup_keep: 1.0,
                        dec_inp_keep: 1.0,
                        dec_oup_keep: 1.0,
                        ext_att_seq: extra_attention_seq,
                        ext_att_len: extra_attention_len,
                        support_seqs: support_tst,
                        classify_tgt: tgt_classify_tst,
                        classify_drop0: 1.0,
                        classify_drop1: 1.0
                    }
                    # Total loss
                    tst_loss = sess.run(loss, fd_tst)
                    tst_total_loss += tst_loss

                    # Classify
                    tst_lossc = sess.run(loss_classify, fd_tst)
                    # standard
                    tst_y_ = sess.run(y_, fd_tst)
                    tst_y_ = np.reshape(tst_y_, [-1])
                    # prediction
                    tst_y = sess.run(y, fd_tst)
                    tst_y = np.reshape(tst_y, [-1])
                    tst_rslt = [int(item > 0.5) for item in tst_y]
                    # train classify information
                    tst_result = mjr.judge_classify(tst_y_, tst_rslt)
                    tst_classify_result += np.array(tst_result)
                    tst_classify_loss += tst_lossc

                    # Extract
                    tst_losse = sess.run(loss_extract, fd_tst)
                    tst_pred = sess.run(prediction, fd_tst)
                    strict_tst, soft_tst = mjr.judge_result(tgt_num_seq_tst, tst_pred)
                    tst_strict_result += np.array(strict_tst)
                    tst_soft_result += np.array(soft_tst)
                    tst_extract_loss += tst_losse

                tst_classify_result /= tst_batch_num
                tst_strict_result /= tst_batch_num
                tst_soft_result /= tst_batch_num
                tst_classify_loss /= tst_batch_num
                tst_extract_loss /= tst_batch_num
                tst_total_loss /= tst_batch_num

                print('  test  loss: {:<10.6}, ({:<10.6}, {:<10.6})'.
                           format(tst_total_loss, tst_classify_loss, tst_extract_loss))
                print('           c: {:<8.4}, {:<8.4}, {:<8.4}'.
                           format(tst_classify_result[0], tst_classify_result[1], tst_classify_result[2]))
                print('        e_st: {:<8.4}, {:<8.4}, {:<8.4}'.
                           format(tst_strict_result[0], tst_strict_result[1], tst_strict_result[2]))
                print('        e_so: {:<8.4}, {:<8.4}, {:<8.4}'.
                           format(tst_soft_result[0], tst_soft_result[1], tst_soft_result[2]))

                flog.write('  test  loss: {:<10.6}, ({:<10.6}, {:<10.6})\n'.
                           format(tst_total_loss, tst_classify_loss, tst_extract_loss))
                flog.write('           c: {:<8.4}, {:<8.4}, {:<8.4}\n'.
                           format(tst_classify_result[0], tst_classify_result[1], tst_classify_result[2]))
                flog.write('        e_st: {:<8.4}, {:<8.4}, {:<8.4}\n'.
                           format(tst_strict_result[0], tst_strict_result[1], tst_strict_result[2]))
                flog.write('        e_so: {:<8.4}, {:<8.4}, {:<8.4}\n'.
                           format(tst_soft_result[0], tst_soft_result[1], tst_soft_result[2]))

                if epoch_num == 0:
                    lowest_loss = dev_extract_loss
                    lowest_count = 0
                else:
                    if lowest_loss < dev_extract_loss:
                        lowest_count += 1
                    else:
                        lowest_loss = dev_extract_loss
                        lowest_count = 0

                print("lowest_loss: {}".format(lowest_loss))
                print("lowest_count: {}".format(lowest_count))

                flog.write("lowest_loss: {}\n".format(lowest_loss))
                flog.write("lowest_count: {}\n".format(lowest_count))
                flog.flush()

                epoch_log_tot_classify[epoch_num] += tst_classify_result
                epoch_log_tot_strict[epoch_num] += tst_strict_result
                epoch_log_tot_soft[epoch_num] += tst_soft_result

                # perform early-stopping
                if lowest_count == 0:
                    early_stop_classify_result = tst_classify_result
                    early_stop_strict_result = tst_strict_result
                    early_stop_soft_result = tst_soft_result
                if epoch_num + 1 == epoch:
                    early_stop_tot_classify += early_stop_classify_result
                    early_stop_tot_strict += early_stop_strict_result
                    early_stop_tot_soft += early_stop_soft_result

                old_dev_soft_result = dev_soft_result
                old_dev_strict_result = dev_strict_result
        flog.close()
        cross_num += 1

    for epoch_iter in range(epoch):
        epoch_log_tot_classify[epoch_iter] /= cross_K
        epoch_log_tot_strict[epoch_iter] /= cross_K
        epoch_log_tot_soft[epoch_iter] /= cross_K

    early_stop_tot_classify /= cross_K
    early_stop_tot_strict /= cross_K
    early_stop_tot_soft /= cross_K

    for epoch_iter in range(epoch):
        print('epoch = {}'.format(epoch_iter))
        f_final.write('epoch = {}\n'.format(epoch_iter))

        a = epoch_log_tot_classify[epoch_iter]
        b = epoch_log_tot_strict[epoch_iter]
        c = epoch_log_tot_soft[epoch_iter]

        print('     c: {:<8.4}, {:<8.4}, {:<8.4}'.format(a[0], a[1], a[2]))
        print('  e_st: {:<8.4}, {:<8.4}, {:<8.4}'.format(b[0], b[1], b[2]))
        print('  e_so: {:<8.4}, {:<8.4}, {:<8.4}'.format(c[0], c[1], c[2]))

        f_final.write('     c: {:<8.4}, {:<8.4}, {:<8.4}\n'.format(a[0], a[1], a[2]))
        f_final.write('  e_st: {:<8.4}, {:<8.4}, {:<8.4}\n'.format(b[0], b[1], b[2]))
        f_final.write('  e_so: {:<8.4}, {:<8.4}, {:<8.4}\n'.format(c[0], c[1], c[2]))

    print('early_stop_result')
    print('     c: {:<8.4}, {:<8.4}, {:<8.4}'.
          format(early_stop_tot_classify[0], early_stop_tot_classify[1], early_stop_tot_classify[2]))
    print('  e_st: {:<8.4}, {:<8.4}, {:<8.4}'.
          format(early_stop_tot_strict[0], early_stop_tot_strict[1], early_stop_tot_strict[2]))
    print('  e_so: {:<8.4}, {:<8.4}, {:<8.4}'.
          format(early_stop_tot_soft[0], early_stop_tot_soft[1], early_stop_tot_soft[2]))

    f_final.write('early_stop_result\n')
    f_final.write('     c: {:<8.4}, {:<8.4}, {:<8.4}\n'.
          format(early_stop_tot_classify[0], early_stop_tot_classify[1], early_stop_tot_classify[2]))
    f_final.write('  e_st: {:<8.4}, {:<8.4}, {:<8.4}\n'.
          format(early_stop_tot_strict[0], early_stop_tot_strict[1], early_stop_tot_strict[2]))
    f_final.write('  e_so: {:<8.4}, {:<8.4}, {:<8.4}\n'.
          format(early_stop_tot_soft[0], early_stop_tot_soft[1], early_stop_tot_soft[2]))

    f_final.close()
