#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : Financial_Index.py
@Author  : Gan Yuyang
@Time    : 2023/4/14 19:41
"""
import datetime
import json
import os
import time

import fake_useragent
import pandas as pd
import requests
from matplotlib import pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
import streamlit as st

def main_finance_index():
    pass


def symbol2cn(symbol):
    df = pd.read_csv('../AllShares/all_share_call.csv')
    lst = list(df['symbol'])
    lst2 = list(df['name'])
    return lst2[lst.index(symbol)]

def text_to_df(text,mode='abs'):
    a = json.loads(text)['data']['list']

    lst_of_dct = []
    mode_dct = {'abs': 0, 'relative': 1}
    for i in a:
        for feature in i.keys():
            i[feature] = i[feature] if type(i[feature]) != list else i[feature][mode_dct.get(mode)]
        lst_of_dct.append(i)

    df = pd.DataFrame(data=lst_of_dct)[::-1]
    suffixes = ['年报', '一季报', '三季报', '中报']
    replacements = ['4', '1', '3', '2']
    df.index = list(df['report_name'].apply(lambda x: x[:4] + '-' + replacements[suffixes.index(x[4:])]))
    df.drop(['report_date', 'ctime', 'report_name'], axis=1, inplace=True)
    # print(df.columns)
    return df





class Sheet():
    def __init__(self, code, year, round=4):
        self.code = code
        self.year = year
        self.headers = {
            'User-Agent': fake_useragent.UserAgent().random,
            'origin': 'https://xueqiu.com',
            'referer': f'https://xueqiu.com/snowman/S/{code}/detail'
        }
        self.session = requests.Session()
        self.ecd = 'utf-8'
        self.session.get("https://xueqiu.com", headers=self.headers)

    def financial_index(self):
        url = f'https://stock.xueqiu.com/v5/stock/finance/cn' \
              fr'/indicator.json?symbol={self.code}&type=all&is_detail=true' \
              f'&count=180&timestamp={1000 * (int(time.mktime(time.strptime(f"{self.year}-12-31", "%Y-%m-%d"))))}'
        for i in range(3):
            try:
                resp = self.session.get(url, headers=self.headers)
                resp.encoding = 'utf-8'
                df_abs = text_to_df(resp.text, mode='abs')
                df_relative = text_to_df(resp.text, mode='relative')
                df_abs.to_csv(f'Financial_Index/abs/{self.code}.csv')
                df_relative.to_csv(f'Financial_Index/relative/{self.code}.csv')
                break
            except Exception:
                continue


def crawl(df, kernal, start):
    for code in tqdm(df.loc[start::kernal, 'symbol']):
        print(code)
        s = Sheet(code, 2023)
        s.financial_index()
        time.sleep(0.1)

def draw_findex(code, mode='abs'):
    df = pd.read_csv(f'Financial_Index/{mode}/{code}.csv', index_col=0)
    lst = ['平均净资产收益率', '每股净利润', '每股经营现金流量', '基本每股收益', '资本公积', '未分配利润每股', '总资产利息支出比率', '净销售率', '毛销售率', '总收入', '营业收入同比增长率', '归属于母公司的净利润', '归属于母公司的净利润同比增长率', '扣除非经常性损益后的归属于母公司的净利润', '扣除非经常性损益后的归属于母公司的净利润同比增长率', '营业外收支净额', '运营资本回报率', '资产负债率', '流动比率', '速动比率', '权益乘数', '权益比率', '股东权益', '经营活动产生的现金流量净额与负债总额之比', '存货周转天数', '应收账款周转天数', '应付账款周转天数', '现金循环周期', '营业周期', '总资本周转率', '存货周转率', '应收账款周转率', '应付账款周转率', '流动资产周转率', '固定资产周转率']
    st.line_chart(df[df.columns[:3]])
    df.columns = lst
    f, ax = plt.subplots(figsize=(10, 9))
    corr = df.draw_findex() * 10
    sns.heatmap(corr, cmap='Blues', linewidths=0.02, annot=False,
                ax=ax, vmin=0, yticklabels=True, xticklabels=True,
                square=False, annot_kws={"fontsize":8})
    plt.title(f'Corr {mode} {code} {symbol2cn(code)}', fontweight='bold')
    plt.subplots_adjust(left=0.2, top=.95)
    plt.rcParams['figure.dpi'] = 300

    plt.show()

"""
营业总收入 total_revenue
每股净利润 np_per_share
平均净资产收益率 avg_roe
每股经营活动产生的现金流量 operate_cash_flow_ps
基本每股收益 basic_eps
"""
if __name__ == '__main__':
    df = pd.read_csv('../AllShares/all_share_call.csv', index_col=0)
    code_lst = df['symbol']

    # for code in tqdm(code_lst[:20]):
    #     corr(code, 'relative')
    draw_findex('SH600519')
    # pool = Pool(processes=15)
    # for i in range(15):
    #     pool.apply_async(func=crawl, args=(df, 15, i+15*(100+30)))
    #
    # pool.close()
    # pool.join()
