#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : test.py
@Author  : Gan Yuyang
@Time    : 2023/4/19 15:50
"""
import pandas as pd

import numpy
import talib
from talib import abstract
close = numpy.random.random(100)
output = talib.abstract

print(output)
dct = talib.get_function_groups()
string = ''
for key in dct.keys():
    string += f'### {key} \n'
    for ind in dct[key]:
        string += f'{ind}\n\n'

with open('../../../Indicators.md', 'w', encoding='utf-8') as f:
    f.write(string)
