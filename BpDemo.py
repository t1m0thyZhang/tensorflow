# -*- coding: utf-8 -*-
# @Time    : 2018/3/7 12:43
# @Author  : timothy
'''
    一个完整的tensorflow demo
    input layer nodes num: 2
    hidden layer nodes num: 3
    output layer nodes num: 1
    数据集：numpy生成
    损失函数：交叉熵  
'''

import tensorflow as tf
from numpy.random import RandomState

batch_size = 8  # 每次学习样本数

# 初始化权重
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 训练数据集
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')  # 输入 n行2列
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')  # 输出 n行1列

# 前向传播
a = tf.matmul(x, w1)
y = tf.matmul(a, w2) # 预测值

# 损失函数
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))  # todo:看交叉熵损失函数公式
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)  # 学习率0.001 优化目标：最小化交叉熵

# 产生模拟数据集
rdm = RandomState(1)
dataset_size = 12800  # 数据集大小
X = rdm.rand(dataset_size, 2)  # 样本输入
Y = [[int(x1+x2 < 1)] for(x1, x2) in X]  # 制造的样本结果 x1+x2<1对应的label为1，否则为0

# 开始训练
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    STEPS = 1600  # 训练次数
    for i in range(STEPS):
        start = i * batch_size
        end = min(start+batch_size, dataset_size)
        # 开始训练
        session.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        # 训练一定时间输出查看交叉熵减小了多少，交叉熵越小误差越小
        if i % 100 == 0:
            total_cross_entroy = session.run(cross_entropy, feed_dict={x:X, y_:Y})
            print(total_cross_entroy)
