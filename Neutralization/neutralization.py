#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : neutralization.py
@Author  : Gan Yuyang
@Time    : 2023/9/7 14:44
"""
import pandas as pd



class Industry:
    def __init__(self):
        self.df: pd.DataFrame = pd.read_csv('D:/Python Projects/Tulip/Neutralization/Industry/snowball/all_share_classified_snowball.csv', index_col=0)

    def get_industries(self) -> list:
        industry_list = list(set(self.df['industry']))
        industry_list.append('其它')
        return industry_list

    def specified(self, industry, mode=1) -> list:
        mode_dict = {1: 'symbol', 2: 'name', 3: 'codename'}
        # print(mode_dict[mode])
        return list(self.df[self.df['industry'] == industry][mode_dict[mode]])

if __name__ == '__main__':

    print(Industry().specified('电机'))
