#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : Sharpe_GD.py
@Author  : Gan Yuyang
@Time    : 2023/6/24 16:54
"""
from timeit import timeit

"""
pytorch 太不自由了
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm, trange
import re
from Sharpe import display_simulated_ef_with_random


def portfolio_annualised_performance(weights, returns, risk_free=0.02):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    annual_returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sha = (annual_returns - risk_free) / std
    return annual_returns, std, sha


def gradient_decent_optim(returns: pd.DataFrame, step_init, n_stocks):
    # 初始化 权重, 步长 等参数
    weights = np.random.rand(n_stocks)
    weights = weights/weights.sum()
    ret, std, sha = portfolio_annualised_performance(weights, returns)
    cur_sha, cur_weights = sha, weights  # 比较项, 前缀cur
    step = step_init
    results = [[ret, std, sha] + list(cur_weights)]

    for epoch in trange(1000):
        """
        res_: 
        [
            [return, std, sha],
            [return, std, sha],
        ]
        """
        _res = []

        for i in range(n_stocks):
            # 计算片对于w[i]的偏导数
            w_for_partial = cur_weights.copy()
            w_for_partial[i] += step
            w_for_partial /= np.sum(w_for_partial)
            ret_i, std_dev_i, sharpe_i = portfolio_annualised_performance(w_for_partial, returns)
            _res.append([ret_i, std_dev_i, sharpe_i])
        _res_sharpe_delta = np.array(_res)[:, 2] - cur_sha
        _max_delta = _res_sharpe_delta.max()
        if _max_delta < 0.0001:
            # stop search if no improvement
            break
        _max = np.abs(_max_delta).max()
        d_weights = _res_sharpe_delta / _max * step if _max < step else _res_sharpe_delta
        new_weights = cur_weights + d_weights
        new_weights[new_weights < 0] = 0.0
        new_weights /= np.sum(new_weights)
        ret_i, std_dev_i, sharpe_i = portfolio_annualised_performance(new_weights, returns)
        if sharpe_i > cur_sha:
            cur_sha, cur_weights = sharpe_i, new_weights
            _r_i = [ret_i, std_dev_i, sharpe_i] + cur_weights.tolist()
            results.append(_r_i)
        else:
            # use smaller delta and search again
            step = step * 0.5
            if step < step_init * 0.1:
                # stop search if increase is less than 10% of initial step
                break
    return np.array(results)

def allocation(results, cols):
    df = pd.DataFrame(results[-1, 3:],index=cols, columns=['allocation'])
    return df


if __name__ == '__main__':
    from Sharpe import equity_ranked_codes, hist_df

    n_stocks = 50
    df = hist_df(equity_ranked_codes()[:n_stocks]).dropna()
    cols= df.columns
    returns = df.pct_change().iloc[1:]
    print(returns.mean()*252)
    # print(a:=gradient_decent_optim(returns, 0.01, n_stocks))
    display_simulated_ef_with_random(returns, 1000*n_stocks, risk_free_rate=0.02, show=False)

    a = gradient_decent_optim(returns, 0.01, len(df.columns))

    print(f'max Sharpe by GD: {a[-1, 2]}')
    print(allocation(a, cols))
    plt.plot(a[:, 1], a[:, 0])
    plt.title(f'Stock num {n_stocks}')
    plt.xlabel("vol")
    plt.ylabel("return")
    plt.show()

    print(portfolio_annualised_performance(a[-1, 3:], returns))
