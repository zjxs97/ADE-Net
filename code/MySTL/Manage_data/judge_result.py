# -*- coding: utf-8 -*-

"""
using the standard bin_seq to evaluate the predict bin_seq
"""

__author__ = 'PKU ChaiZ'
import numpy as np


# directly used for judge_result()
def _change_format(gold_standard, prediction):
    """
    turn each seq into (start, end)-pair form
    :param gold_standard: the gold standard bin_seq
    :param prediction: the predicted bin_seq
    :return: the changed form
    """
    gold_pairs = []
    for seq in gold_standard:
        posit = 0
        this_seq_pairs = []
        while posit < len(seq):
            if (seq[posit]) == 1:
                num1 = posit
                posit += 1
                while (posit < len(seq)) and (seq[posit] == 2):
                    posit += 1
                num2 = posit
                this_seq_pairs.append((num1, num2))
            else:
                posit += 1
        gold_pairs.append(this_seq_pairs)

    pred_pairs = []
    for seq in prediction:
        posit = 0
        this_seq_pairs = []
        while posit < len(seq):
            if (seq[posit]) == 1:
                num1 = posit
                posit += 1
                while (posit < len(seq)) and (seq[posit] == 2):
                    posit += 1
                num2 = posit
                this_seq_pairs.append((num1, num2))
            else:
                posit += 1
        pred_pairs.append(this_seq_pairs)

    return gold_pairs, pred_pairs


# directly used for judge_result()
def _judge_strict(gold_pairs, pred_pairs):
    """
    :param gold_pairs: gold standard in (start, end)-pair form
    :param pred_pairs: prediction in (start, end)-pair form
    :return: F1, precision and recall using the strict standard
    """
    TP = pred_all = gold_all = 0
    for index in range(len(gold_pairs)):
        pred_all += len(pred_pairs[index])
        gold_all += len(gold_pairs[index])
        for pairs in pred_pairs[index]:
            if pairs in gold_pairs[index]:
                TP += 1
    if TP == 0:
        F1 = p = r = 0
    else:
        p = TP / pred_all
        r = TP / gold_all
        F1 = (2 * p * r) / (p + r)
    return float(F1), float(p), float(r)


# directly used for judge_result()
def _judge_soft(gold_pairs, pred_pairs):
    """
    :param gold_pairs: gold standard in (start, end)-pair form
    :param pred_pairs: prediction in (start, end)-pair form
    :return: F1, precision and recall using the soft standard
    """
    TP = pred_all = gold_all = 0
    for index in range(len(gold_pairs)):
        gold_set = set()
        pred_set = set()
        for item in gold_pairs[index]:
            for num in range(item[0], item[1]):
                gold_set.add(num)
        for item in pred_pairs[index]:
            for num in range(item[0], item[1]):
                pred_set.add(num)
        set_both = gold_set & pred_set
        pred_all += len(pred_set)
        gold_all += len(gold_set)
        TP += len(set_both)
    if TP == 0:
        p = r = F1 = 0
    else:
        p = TP / pred_all
        r = TP / gold_all
        F1 = (2 * p * r) / (p + r)
    return float(F1), float(p), float(r)


def judge_result(gold_standard, prediction):
    """
    :param gold_standard: standard bin_seq
    :param prediction: predicted bin_seq
    :return: the evaluation using strict and soft judge method
    """
    gold_pairs, pred_pairs = _change_format(gold_standard, prediction)

    strict_val = _judge_strict(gold_pairs, pred_pairs)
    soft_val = _judge_soft(gold_pairs, pred_pairs)

    return strict_val, soft_val


def judge_classify(standard, prediction):
    """
    :param standard: the standard answer
    :param prediction: the predicted answer
    :return: the F1, Precesion and the Recall
    """
    label = np.array(standard, dtype=np.int32)
    logit = np.array(prediction, dtype=np.int32)
    TP = np.sum(label * logit)
    if TP == 0:
        F1 = p = r = 0
    else:
        p = TP / np.sum(logit)
        r = TP / np.sum(label)
        F1 = (2 * p * r) / (p + r)
    return float(F1), float(p), float(r)
