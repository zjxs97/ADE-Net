# -*- coding: utf-8 -*-

"""
different kinds of loss functions
"""

__author__ = 'PKU ChaiZ'

import tensorflow as tf


def get_classify_layers(dense0, drop0, drop1):
    # prediction
    dense0_drop = tf.nn.dropout(dense0, drop0)
    dense1 = tf.contrib.layers.fully_connected(
        dense0_drop, 20,
    )
    dense1_drop = tf.nn.dropout(dense1, drop1)
    dense2 = tf.contrib.layers.fully_connected(
        dense1_drop, 1,
        activation_fn=tf.nn.sigmoid,
    )
    y = dense2

    return y
