#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : Snowball_classity.py
@Author  : Gan Yuyang
@Time    : 2023/4/18 19:46
"""
import json
import re

import fake_useragent
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def json2lst(text):
    text = json.loads(text)
    return text['data']['list']

headers = {
    'User-Agent': fake_useragent.UserAgent().random,
}

def fields_code_name():
    resp = requests.get(r'https://xueqiu.com/hq#', headers=headers)
    resp.encoding = 'utf-8'
    content = resp.text
    # with open('2.html', 'w', encoding='utf-8') as f:
    #     f.write(resp.text)
    # with open('2.html', 'r', encoding='utf-8') as f:
    #     content = f.read()
    name_lst = []
    code_lst = []
    body = BeautifulSoup(content, 'lxml')
    parts = body.find('div', {'class': 'third-nav'}).findAll('ul')

    for m in parts:
        code_lst.extend(re.findall(r'data-level2code="(.*?)"', str(m)))
        name_lst.extend(re.findall(r'title="(.*?)"', str(m)))
        # for i in f:
        #
        #     code_lst.append(i.findNext('a').get('data-level2code'))
        #     name_lst.append(i.findNext('a').get('title'))
    return code_lst, name_lst

def df_classify(df:pd.DataFrame, save=False):
    df['industry'] = df.apply(lambda x: '', axis=1)
    session = requests.session()
    session.get('https://xueqiu.com/', headers=headers)
    code_lst, name_lst = fields_code_name()
    start = 0
    for code, name in tqdm(zip(code_lst[start:], name_lst[start:]), total=len(code_lst[start:])):
        # print(code, name)
        resp = session.get(f'https://stock.xueqiu.com/v5/stock/screener/quote/list.json?page=1&size=90&order=desc&order_by=percent&exchange=CN&market=CN&ind_code={code}', headers=headers)
        resp.encoding = 'utf-8'
        # print(resp.url)
        # with open('1.txt', 'w', encoding='utf-8') as f:
        #     f.write(resp.text)
        # print(content)
        lst_of_dct = json2lst(resp.text)
        for info_dict in lst_of_dct:
            symbol = info_dict['symbol']
            df.loc[df['symbol'] == symbol, 'industry'] = name
    if save:
        df.to_csv('all_share_classified_snowball.csv', encoding='utf-8')

    return df

if __name__ == '__main__':
    df = pd.read_csv(r'../../../AllShares/all_share_call.csv', index_col=0)
    print(len(fields_code_name()[0]))
    # [print(i) for i in fields_code_name()[0]]
    df_classify(df, save=True)

