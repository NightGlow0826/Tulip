#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : collect_share_list.py
@Author  : Gan Yuyang
@Time    : 2023/4/14 23:15
"""
import datetime
import json
import time

import fake_useragent
import pandas as pd
import requests
from tqdm import trange

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

# static

headers = {
    'User-Agent': fake_useragent.UserAgent().random
}
session = requests.Session()
session.get('https://xueqiu.com/', headers=headers)
lst_of_symbols = []
for i in trange(55):
    url = f'https://stock.xueqiu.com/v5/stock/screener/quote/list.json?page={i+1}&size=90&order=desc&orderby=percent&order_by=percent&market=CN&type=sh_sz'

    resp = session.get(url, headers=headers)
    resp.encoding = 'utf-8'
    dct = json.loads(resp.text)
    # print(dct)
    try:
        lst_of_symbols.extend(dct['data']['list'])
    except Exception:
        break
    # quit()

df = pd.DataFrame(lst_of_symbols)
df.sort_values(by=['symbol'], inplace=True)
df.to_csv('all_share_basic.csv', encoding='utf-8')
