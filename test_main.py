# -*- coding: utf-8 -*-
# @Time    : 2018/3/17 16:07
# @Author  : timothy

import numpy as np
import pandas as pd

def create_dict():
    d= {'k1':'v1','k2':'v2','k3':'v3'}
    # print(dct)
    for k, v in d.items():
        print(k)
        print(v)

def test():
    string = 'aaa'
    print('hello main\n a')

def main():
    print('hello main\n a')


def test_numpy():
    # a = [[1,2,3],[4,5,6],[7,8,9]]
    a = np.arange(9).reshape(3, 3)
    print(type(a))
    print(a)


def test_dataframe():
    dates = pd.date_range('20130101', periods=6)
    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))

    print(df)

if __name__ == '__main__':
    test_dataframe()
    # test_numpy()
    # main()
    # create_dict()
    # test()