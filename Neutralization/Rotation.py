#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : Rotation.py
@Author  : Gan Yuyang
@Time    : 2023/9/7 16:56
"""
import numpy as np
import pandas as pd

from neutralization import Industry
from stockquery.query import Hist_data
from Stochastic.distri import Portfolio, Farray

industry = Industry()
ind_list = industry.get_industries()

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

# st_list = ['计算机设备', '软件开发', '塑料', '化学制药']
# st_list = ['化学制药']
# st_list = ['化学制药']

for (pos, ind) in enumerate(ind_list):
    spe_list = industry.specified(ind)
    if a:=len(spe_list) <= 10:
        print(f'industry {ind} Skipped')
        continue
    print(ind)
    spe_list = [str_[2:] for str_ in spe_list]

    his = Hist_data(spe_list)
    df = his.simple_close()
    df_cat = pd.concat([df, his.get_index('sz')['close']], axis=1)

    # 缺失超过 20% 就删除
    nan_ratio = df_cat.isnull().sum() / len(df)
    # 找到满足条件的列
    columns_to_drop = nan_ratio[nan_ratio > 0.2].index
    # 删除满足条件的列
    df_cat = df_cat.drop(columns_to_drop, axis=1)
    # df_cat.dropna(how='all', axis=1, inplace=True)
    #
    df_cat.dropna(how='any', axis=0, inplace=True)

    ret = df_cat.pct_change()
    ret.dropna(inplace=True)
    # ret.to_csv('phar_ret.csv')
    # ret = pd.read_csv('phar_ret.csv', index_col=0)
    ret = ret.loc[:, (ret >= -1).all()] # 有的股票能超跌 100%

    portf = Portfolio(array=ret.iloc[:, :-1])
    portf.min_std_lim_math(optim=True, only_upper=False, title=ind, sharpe_color=False, weight_inplace_mode=2)
    seq_log_ret = np.log(np.array(portf.seq_return()) + 1)
    far = Farray(seq_log_ret)
    far.qq(title=ind)



