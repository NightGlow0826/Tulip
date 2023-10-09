#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : 2023-10-9.py
@Author  : Gan Yuyang
@Time    : 2023/10/9 16:00
"""
from distri import Portfolio, Farray, get_rets
import pandas as pd

def frontier():
    df = pd.read_excel(r"C:\Users\NightGlow\Documents\Tencent Files\3218275879\FileRecv\return_data.xls",
                       index_col=0).iloc[:, :-1]
    # already rets
    # print(df)

    port = Portfolio(df)
    port.min_std_lim_math(optim=False, sharpe_color=False)
    print(port.weight)

frontier()

