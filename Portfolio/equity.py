#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : equity.py
@Author  : Gan Yuyang
@Time    : 2023/6/16 19:31
"""
import re

import bs4

with open('contet.html', 'r', encoding='utf-8') as f:
    content = f.read()
    codes = re.findall('<div class="td-cell-box">(.*?)<', content)
    print(codes)