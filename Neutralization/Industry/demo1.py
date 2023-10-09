#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : demo1.py
@Author  : Gan Yuyang
@Time    : 2023/4/24 16:38
"""
from datetime import datetime

import numpy as np
import pandas as pd
import tushare as ts
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
import talib
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import trange

"""
This aims to first compare a single stock to the same industry
"""
ts.set_token('0d137339fa0ff2e8fb8d80e2c3fb24400513ddd9c1d18b947d313fd3')
pro = ts.pro_api()


class Stock:
    def __init__(self, code, name, industry):
        self.code = code
        self.name = name
        self.industry = industry

        today = datetime.today().strftime('%Y%m%d')
        # previous = (parse(today) + relativedelta(days=-10)).strftime('%Y%m%d')
        self.data = pro.query('daily', ts_code=f'{code[2:]}.{code[:2]}', end_date=today)[
                    ::-1].reset_index().drop(['index', 'ts_code'], axis=1)

    def simple_normalization(self):
        # 所有价格除以第一天的开盘价
        first_open = self.data.iloc[0, 1]
        print(first_open)
        self.data[['open', 'high', 'low', 'close']] = self.data[['open', 'high', 'low', 'close']].apply(
            lambda x: x / first_open)

        return self.data

    def add_nday_return(self, n=2):
        self.data[f'{n}_re'] = self.data['close'].pct_change(periods=-n)


    def ma_add(self, period_lst):
        s = talib


    def draw_corr(self):
        df = self.data.drop(['trade_date', 'open', 'high', 'low', 'pre_close','change', 'pct_chg'], axis=1)
        print(df)
        f, ax = plt.subplots(figsize=(12, 9))

        corr = df.draw_findex('kendall')

        sns.heatmap(corr, cmap='viridis', linewidths=0.02, ax=ax, vmin=0,
                    annot=True)
        plt.rcParams['font.sans-serif'] = ['SimHei']

        plt.title(f'Corr {self.code} {self.name}')
        plt.rcParams['figure.dpi'] = 500
        plt.subplots_adjust(left=0.2, top=.95)

        plt.show()

def c_df(mode='snowball'):
    return pd.read_csv(f'{mode}/all_share_classified_{mode}.csv', index_col=0)


def i_df(df: pd.DataFrame, industry):
    # 返回给定行业的df
    return df[df['industry'] == industry].reset_index().drop(['index'], axis=1).dropna(axis=0)


def lst_of_stockclass(classified_df, industry):
    lst = [Stock(
        code=classified_df.loc[i, 'symbol'],
        name=classified_df.loc[i, 'name'],
        industry=industry
    ) for i in trange(len(classified_df)-20)]

    return lst


if __name__ == '__main__':
    df = c_df()
    i_lst = list(df['industry'].drop_duplicates().dropna())
    print(i_lst)

    ind = '普钢'
    print(i_df(df, ind))
    cs = lst_of_stockclass(i_df(df, ind), ind)
    print(len(cs))
    for c in cs[:10]:
        c.simple_normalization()
        for n in [1, 2, 7, 30, 60, 120]:
            c.add_nday_return(n)

        c.draw_corr()
