# -*- coding: utf-8 -*-
# @Time    : 2018/3/26 16:40
# @Author  : timothy
'''
 mnist手写数字识别(单层神经网络)
 输入：像素为【28*28】的图片
 输出：数字【0-9】的预测值
 隐层神经元数目：500
 方法：使用滑动平均的随机梯度下降
 损失函数：交叉熵加L2正则项
 其它：学习率指数衰减
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 初始化常量
INPUT_NODE = 784   # 28*28
OUTPUT_NODE = 10  # 0-9
LAYRR1_NODE = 500  # 隐层神经元数
BATCH_SIZE = 100   # 1就是随机梯度下降，整个数据集就是梯度下降
LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 正则化项在损失函数里的权重
TRAINING_STEP = 5000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


# 前向传播
def inference(input_tensor, avg_class, w1, b1, w2, b2):
    if avg_class is None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, w1)+b1)  # y = wx + b
        return tf.matmul(layer1, w2)+b2
    else:
        # 随机梯度下降，用平均权重和偏置代替
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(w1)) + avg_class.average(b1))
        return tf.matmul(layer1, avg_class.average(w2)) + avg_class.average(b2)


# 开始训练
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')  # 输入图片
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')  # 图片标签0-9

    # 随机初始化输入层到隐层之间的w，和隐层的b
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYRR1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYRR1_NODE]))
    # 随机初始化隐层和输出层之间的w，和输出层的b
    weights2 = tf.Variable(tf.truncated_normal([LAYRR1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y = inference(x, None, weights1, biases1, weights2, biases2)  # 不采用随机梯度下降学习出的答案
    global_step = tf.Variable(0, trainable=False)  # 训练轮数 该参数不可训练
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 定义损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)  # 交叉熵
    cross_entropy_mean = tf.reduce_mean(cross_entropy)  # 当前batch交叉熵均值

    # 引入L2正则
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)

    # 引入正则项后的损失函数
    loss = cross_entropy_mean + regularization

    # 设定学习率指数衰减
    iter_times = mnist.train.num_examples / BATCH_SIZE
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, iter_times, LEARNING_RATE_DECAY)

    # 执行梯度下降
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    # 更新滑动平均值
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 计算准确率
    correction_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))

    # 开始训练
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        for i in range(TRAINING_STEP):
            if i % 1000 == 0:
                validate_acc = session.run(accuracy, feed_dict=validate_feed)
                print('after training for %d steps, training accuracy is %g' % (i + 1000, validate_acc))
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            session.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = session.run(accuracy, feed_dict=test_feed)
        print('testing accuracy is %g' % (test_acc))


def main():
    mnist = input_data.read_data_sets('resource/', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    main()
