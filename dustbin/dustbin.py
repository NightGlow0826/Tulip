#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : dustbin.py
@Author  : Gan Yuyang
@Time    : 2023/4/13 22:08
"""
import time
import pandas as pd
import pyperclip
import re
import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']


def a():
    while 1:
        a = input('sth: ')
        lst = a.split('\r')
        a = re.findall(r'.*?(\d.*)', a)[0].strip()
        a = re.sub(r'． ', '..', a)
        a = re.sub(r'\. ', '..', a)

        a = re.sub(r' ', '\n', a)
        pyperclip.copy(a)
        print(a)
        print()


def b(start=1):
    while 1:
        a = input('sth: ')
        strlst = []
        lst = a.split('\n')
        print(lst)
        for i in lst:
            a = re.sub(r'． ', '..', i)
            a = re.sub(r'\. ', '..', a)
            a = a.split()[start:]
            a = '\t'.join(a)
            strlst.append(a)
        string1 = '\n'.join(strlst)
        pyperclip.copy(string1)
        print(string1)
        print()
        time.sleep(1)


# 2006 28048656 2287254 12316574 13444828 20615
# b()
def st_mutiline():
    with st.form('line'):
        start = st.selectbox('start', [i for i in range(10)])
        end = st.selectbox('end', [i for i in range(1, 50)])
        a = st.text_area('input')
        fsb = st.form_submit_button('submit')
        strlst = []
        lst = a.split('\n')
        if fsb:
            for i in lst:
                a = re.sub(r'． ', '..', i)
                a = re.sub(r'\. ', '..', a)
                a = a.split()[start:end]
                a = '\t'.join(a)
                strlst.append(a)
            string1 = '\n'.join(strlst)
            pyperclip.copy(string1)
            print(string1)
            print()
            time.sleep(1)
            st.write(string1)


def draw_corr():
    df = pd.read_excel(r'C:\Users\NightGlow\Desktop\成都市经济统计.xlsx', index_col=0).iloc[:-1]
    print(df)
    f, ax = plt.subplots(figsize=(12, 9))

    corr = df.draw_findex()
    cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)

    sns.heatmap(corr, cmap='viridis', linewidths=0.02, ax=ax, vmin=0.85,
                )
    plt.title('成都市各经济指标相关系数')
    plt.rcParams['figure.dpi'] = 300
    plt.subplots_adjust(left=0.2, top=.95)

    plt.show()


def draw_line():
    df = pd.read_excel(r'C:\Users\NightGlow\Desktop\成都市经济统计.xlsx', index_col=0).iloc[:-1]
    print(df.columns)
    print(df.index)
    # print(df)
    fig, ax = plt.subplots(figsize = (12, 9))
    print((df['年份']))
    ax.plot(df['年份'], df['地区生产总值（千万元）'], '*-', label='地区生产总值（千万元）')
    ax.plot(df['年份'], df['人均生产总值（元）'], '*-', label='人均生产总值（元）')
    ax.grid()
    a = ['0' + str(i) for i in range(3, 10)]
    a.extend([str(i) for i in range(10, 22)])
    ax.set_xticks(df['年份'], a, rotation=60)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    draw_corr()
