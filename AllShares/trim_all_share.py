#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : trim_all_share.py
@Author  : Gan Yuyang
@Time    : 2023/4/14 23:36
"""

# static
import pandas as pd

df = pd.read_csv('all_share_basic.csv', encoding='utf-8', index_col=0)
df.sort_values(by=['symbol'], inplace=True)
df.index = [i for i in range(len(df))]
df['codename'] = df.apply(lambda x: x['symbol'] + ' ' + x['name'], axis=1)

df.to_csv('all_share_basic.csv', encoding='utf-8') # 保留一些基本信息
df[['symbol', 'name','codename']].to_csv('all_share_call.csv', encoding='utf-8') # 只保留名字和编号