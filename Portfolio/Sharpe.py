#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : Sharpe.py
@Author  : Gan Yuyang
@Time    : 2023/6/16
"""

import akshare as ak
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import re

# sharpe ratio

def equity_ranked_codes():
    with open('D:\Python Projects\Tulip\Portfolio\contet.html', 'r', encoding='utf-8') as f:
        content = f.read()
        codes = re.findall('<div class="td-cell-box">(.*?)<', content)
    return codes

# copy 来的权重修正函数
def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns @ weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns


# copy 来的权重生成函数
def random_portfolios(num_stocks, num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((num_stocks, num_portfolios))
    weights_record = []
    for i in tqdm(range(num_portfolios)):
        weights = np.random.random(num_stocks)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_std_dev  # 波动率目标
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev  # 夏普比率目标
    return results, weights_record


# copy 来的 收益率边界画图函数
def display_simulated_ef_with_random(all_df,num_portfolios, risk_free_rate, show=True):
    mean_returns = all_df.mean()
    cov_matrix = all_df.cov()
    num_stocks = len(all_df.columns)
    results, weights = random_portfolios(num_stocks, num_portfolios, mean_returns, cov_matrix, risk_free_rate)

    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index=all_df.columns, columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx], index=all_df.columns, columns=['allocation'])
    min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    print("-" * 80)
    print(f'Sharpe Ratio: {results[2, max_sharpe_idx]}')
    print("-" * 80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp, 2))
    print("Annualised Volatility:", round(sdp, 2))
    print("\n")
    print(max_sharpe_allocation)

    print("-" * 80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min, 2))
    print("Annualised Volatility:", round(sdp_min, 2))
    print("\n")
    print(min_vol_allocation)

    plt.figure(figsize=(10, 7))
    plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='YlGnBu', marker='o', s=10, alpha=0.1)
    plt.colorbar()
    plt.scatter(sdp, rp, marker='*', color='r', s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min, rp_min, marker='*', color='g', s=500, label='Minimum volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns_tensor')
    plt.legend(labelspacing=0.8)
    plt.show() if show else ...
    return max_sharpe_allocation, min_vol_allocation


def hist_df(stocklist):
    histdata = {}
    for i in stocklist:
        try:
            histdata[i] = ak.stock_zh_a_hist(symbol=i, period="daily",
                                       start_date="20150101", end_date='20230801',
                                       adjust="qfq")['收盘']
            # print(histdata)

        except Exception as e:
            print(i, e)


    for i in histdata.values():
        i.columns = ['close']


    # 理想的数据结构
    df = pd.DataFrame(data=histdata)
    return df


stocklist = ['600519', '601398', '601857', '000300']
# stocklist = equity_ranked_codes()[:10]
df= hist_df(stocklist)



# 参数设置
returns = df.pct_change().iloc[1:]
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = int(10000 * (num_stocks := len(stocklist)))
risk_free_rate = 0.02

# plt.show()
if __name__ == '__main__':

    # max_sharpe_alloc, min_vol_alloc = display_simulated_ef_with_random(returns,  num_portfolios, risk_free_rate)

    # max_sharpe_alloc.to_csv('portfolio_MC.csv')
    print(hist_df(stocklist))