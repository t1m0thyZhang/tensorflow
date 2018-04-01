# -*- coding: utf-8 -*-
# @Time    : 2018/4/1 13:14
# @Author  : timothy


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_cnn_inference as mci
import numpy as np


# 初始化常量
BATCH_SIZE = 100   # 1就是随机梯度下降，整个数据集就是梯度下降
LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 正则化项在损失函数里的权重
TRAINING_STEP = 5000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


# 开始训练
def train(mnist):
    x = tf.placeholder(tf.float32, [BATCH_SIZE, mci.IMAGE_SIZE, mci.IMAGE_SIZE, mci.NUM_CHANNELS], name='x-input')  # 输入图片
    y_ = tf.placeholder(tf.float32, [None, mci.OUTPUT_NODE], name='y-input')  # 图片标签0-9

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)  # 正则化
    y = mci.inference(x, None, regularizer)  # 不采用随机梯度下降学习出的答案
    global_step = tf.Variable(0, trainable=False)  # 训练轮数 该参数不可训练
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 定义损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)  # 交叉熵
    cross_entropy_mean = tf.reduce_mean(cross_entropy)  # 当前batch交叉熵均值

    # 引入正则项后的损失函数
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 设定学习率指数衰减
    iter_times = mnist.train.num_examples / BATCH_SIZE
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, iter_times, LEARNING_RATE_DECAY)

    # 执行梯度下降
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    # 更新滑动平均值
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 计算准确率
    correction_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))

    # 保存训练结果
    saver = tf.train.Saver()

    # 开始训练
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        # 准备测试数据
        x_test, y_test = mnist.test.next_batch(BATCH_SIZE)
        x_test_reshaped = np.reshape(x_test, (BATCH_SIZE, mci.IMAGE_SIZE, mci.IMAGE_SIZE, mci.NUM_CHANNELS))
        test_feed = {x: x_test_reshaped, y_: y_test}
        # 准备训练数据
        x_train, y_train = mnist.train.next_batch(BATCH_SIZE)
        x_train_reshaped = np.reshape(x_train, (BATCH_SIZE, mci.IMAGE_SIZE, mci.IMAGE_SIZE, mci.NUM_CHANNELS))
        train_feed = {x: x_train_reshaped, y_: y_train}
        # 开始训练
        for i in range(TRAINING_STEP):
            # 每次训练BATCH_SIZE张图片
            session.run(train_op, feed_dict=train_feed)
            # 每一百步进行一个测试
            if i % 100 == 0:
                test_acc = session.run(accuracy, feed_dict=test_feed)
                print('after training for %d steps, training accuracy is %g' % (i + 100, test_acc))


        # save_dir = 'result/mnist_cnn.ckpt'
        # saver.save(session, save_dir)  # 保存训练结果
        # # saver.restore(session, save_dir)  # 恢复上次的训练结果
        # test_acc = session.run(accuracy, feed_dict=test_feed)
        # print('testing accuracy is %g' % (test_acc))
        # tf.summary.FileWriter('cnn_logs', session.graph)


def main():
    mnist = input_data.read_data_sets('resource/', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    main()
