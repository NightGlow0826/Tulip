#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : Sharpe_BP.py
@Author  : Gan Yuyang
@Time    : 2023/6/24 14:51
"""
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import trange

"""
This is an attempt aiming to use BP net to calculate the weights of the minimum
sharpe ratio portfolio
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from Sharpe import equity_ranked_codes, hist_df

num_stocks = 4
df = hist_df(equity_ranked_codes()[:num_stocks])
# print(df.columns)
returns = df.pct_change().iloc[1:]
risk_free_rate = 0.02

print(returns)


def sharpe_ratio(returns, weights, risk_free=0.02):
    weighted_return = returns @ weights
    print(weighted_return)
    ave = np.mean(weighted_return)
    std = np.std(weighted_return)
    return (ave - risk_free) / std


if __name__ == '__main__':
    weights = np.random.uniform(0, 1, num_stocks)
    weights = weights / weights.sum()
    print(sharpe_ratio(returns, weights))
