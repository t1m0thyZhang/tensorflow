# -*- coding: utf-8 -*-
# @Time    : 2018/3/31 15:08
# @Author  : timothy
'''
    使用CNN进行mnist手写数字识别
    输入：28*28*1
    卷积层：卷积核大小：5*5，深度为32，步长为1，用全0填充   28*28*1 -》 28*28*32
    池化层：过滤器大小：2*2 步长2   28*28*32 -》14*14*32
    卷积层：卷积核大小：5*5，深度为64，步长为1，用全0填充  14*14*32 -》14*14*64
    池化层：过滤器大小：2*2 步长2   14*14*64 -》7*7*64
    全连接层：512个节点   7*7*64 -》512
    全连接层：10个节点    512 -》10
    输出：使用softmax
'''

import tensorflow as tf


INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1  # 通道数，mnist是黑白的因此是1
NUM_LABLES = 10

# 卷积层1
CONV1_DEEP = 32
CONV1_SIZE = 5

# 卷积层2
CONV2_DEEP = 64
CONV2_SIZE = 5

# 全连接层数目
FC_SIZE = 512


def inference(input_tensor, train, regularizer):

    # 卷积层1  28*28*1 -》28*28*32
    # 变量命名空间，这样weight和biases变量和后续层就不会重复
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable('weight', [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],\
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('biases', [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        # strides第一，四个参数只能为1，中间两参数表示在长宽两方向的步长为1；padding='SAME'全0填充
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    #  池化层1  28*28*32 -》 14*14*32
    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #  卷积层2  14*14*32 -》 14*14*64
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable('weight', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],\
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('biases', [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        # strides第一，四个参数只能为1，中间两参数表示在长宽两方向的步长为1；padding='SAME'全0填充
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    #  池化层2  14*14*64 -》7*7*64
    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 将池化层输出 【7*7*64】 的3维张量拉成一个向量
    pool_shape = pool2.get_shape().as_list()  #
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    #  全连接层1
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable('weight', [nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('biases', [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)  # 避免过拟合

    #  全连接层2
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable('weight', [FC_SIZE, NUM_LABLES],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable('biases', [NUM_LABLES], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit