# -*- coding: utf-8 -*-
# @Time    : 2018/3/31 15:35
# @Author  : timothy
# 数据持久化

import tensorflow as tf


def train():
    v1 = tf.Variable(tf.constant(1.0), name='v1')
    v2 = tf.Variable(tf.constant(2.0), name='v2')
    result = v1 + v2
    saver = tf.train.Saver()
    dir = 'result/trained.ckpt'  # 保存路径
    # 开始计算
    with tf.Session() as session:
        # 第一次使用保存
        # session.run(tf.global_variables_initializer())
        # saver.save(session, dir)
        # 第二次直接使用即可
        saver.restore(session, dir)
        print(session.run(result))


def main():
    train()


if __name__ == '__main__':
    main()