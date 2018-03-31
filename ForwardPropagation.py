# -*- coding: utf-8 -*-
# @Time    : 2018/3/6 21:16
# @Author  : timothy
'''
    前向传播
    该神经网络输入2个节点，隐层3个节点，输出1个节点
'''

import tensorflow as tf

# 随机初始化权重
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.constant([[0.7, 0.9]])  # 输入2个节点
a = tf.matmul(x, w1)  # 中间层结果
y = tf.matmul(a, w2)  # 输出结果

# 用with自动管理session关闭
with tf.Session() as session:
    # 必须要初始化，前面的w1只保存了计算的方式，这里才真正计算
    session.run(w1.initializer)
    session.run(w2.initializer)
    # 变量多了用全部初始化方法
    #session.run(tf.global_variables_initializer())
    print(session.run(y))