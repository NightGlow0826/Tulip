#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : test.py
@Author  : Gan Yuyang
@Time    : 2023/9/7 19:14
"""

a = [1, 2]
b = [1, 2]
c = [1, 2]
import numpy as np
def f(*args):
    l = np.array(args)
    print(l.argmax(axis=1))
    print(l)
    return l

print(*f(a, b, c))
