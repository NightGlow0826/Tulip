#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : temp.py
@Author  : Gan Yuyang
@Time    : 2023/4/23 15:26
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd


pd.set_option('display.max_rows', 1000)


from tqdm import *

from pyecharts import options as opts
from pyecharts.charts import Map, Page, Bar, Line, Kline, Grid
from pyecharts.commons.utils import JsCode
import tushare as ts

from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
from datetime import datetime

ts.set_token('0d137339fa0ff2e8fb8d80e2c3fb24400513ddd9c1d18b947d313fd3')
pro = ts.pro_api()
today = datetime.today().strftime('%Y%m%d')
previous = (parse(today) + relativedelta(days=-4)).strftime('%Y%m%d')
df = pro.query('daily', ts_code='600372.SH', start_date=previous, end_date=today)
df.set_index('ts_code', inplace=True)
print(df.columns)
