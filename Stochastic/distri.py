#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : distri.py
@Author  : Gan Yuyang
@Time    : 2023/8/26 16:27
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import tqdm
from scipy import stats
from tqdm import trange
import akshare as ak


def get_rets(mode=1, stocks=10):
    """

    :param mode: 1-> dataframe, 2-> ndarray
    :return:
    """
    if mode == 1:
        from Portfolio.Sharpe import hist_df, equity_ranked_codes

        stocklist = equity_ranked_codes()[:stocks]
        df = hist_df(stocklist)
        rets: pd.DataFrame = df.pct_change().iloc[1:]

        return rets
    elif mode == 2:
        rets = (np.random.normal(0, 2, 10000))
        return rets


def ret_period(cur_rate, cur, tar):
    return (1 + cur_rate) ** (tar / cur) - 1


def get_index(start='2015-01-01', end='2023-08-01'):
    stock_zh_index_daily_df = ak.stock_zh_index_daily(symbol="sh000300")
    stock_zh_index_daily_df['date'] = pd.to_datetime(stock_zh_index_daily_df['date']).dt.strftime('%Y-%m-%d')
    stock_zh_index_daily_df.set_index('date', inplace=True)
    stock_zh_index_daily_df = stock_zh_index_daily_df['close'].pct_change().dropna()
    return stock_zh_index_daily_df


class Farray:
    def __init__(self, array, risk_free=0., period=252):
        self.array: pd.DataFrame | np.ndarray = array
        ...
        self.period = period
        self.risk_free = risk_free
        if isinstance(self.array, pd.DataFrame):
            self.len = self.array.shape[1]
            self.date_len = self.array.shape[0]
            self.weight: np.ndarray = np.random.random(self.len)
            self.weight /= self.weight.sum()
            # self.weight = self.weight.reshape(len(self.weight), -1)

    def __len__(self):
        return self.array.shape[1]

    def __str__(self):
        return "This class receives a return of dataframe or ndarray, "

    @property
    def mean_return(self):
        daily_mean = ((1 + self.array).prod()) ** (1 / self.array.shape[0]) - 1
        return (daily_mean + 1) ** np.sqrt(self.period) - 1

    @property
    def mean_std(self):
        return self.array.std() * np.sqrt(self.period)

    def log_return(self):
        return np.log(self.array + 1)

    @property
    def cov(self):
        # i, j 为 i, j 的协方差
        return np.cov(self.array, rowvar=False) * self.period

    def moment(self, order):
        """
        计算数列的矩
        :param order: 阶数
        :return:
        """
        """
        order=3, 偏度 skewness
        order=4, 峰度 kurtosis
        """
        demeaned = self.array - np.mean(self.array)
        exp_demeaned = np.mean(demeaned ** order)  # 分子
        denominator = np.std(self.array, ddof=1) ** order  # 分母
        return exp_demeaned / denominator

    def draw_hist(self, show_pic=None, bins=10, quickshow=False, norm_bench=False):
        if show_pic is None:
            show_pic = self.array
        plt.hist(show_pic, bins=bins)

        # if norm_bench:
        #     temp = Farray(show_pic)
        #     mu = temp.moment(order=1)  # 均值
        #     sigma = temp.moment(order=2)  # 标准差
        #     x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)  # x轴的取值范围
        #     y = 200*np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))  # 计算概率密度函数

        if quickshow:
            plt.show()

    def normal_check(self, level=0.01):

        def is_normal(r, level=level) -> bool:
            test_statistic, p_value = scipy.stats.jarque_bera(r)
            return p_value > level

        if isinstance(self.array, pd.DataFrame):
            return self.array.aggregate(is_normal)
        else:
            return is_normal(self.array, level)

    def qq(self, title=''):
        stats.probplot(self.array, dist="norm", plot=plt)
        plt.xlabel('')
        plt.ylabel('')
        plt.title(title)
        plt.show()

    def var_historic(self, level=0.05):
        # 日均, 其余天数为 * sqrt(day)
        def var_historic_sig(r, level=0.05, period=self.period):
            """
            给定 dataframe|array 计算var
            显著性水平计算时写成 e.g., 5, 而非 5%
            :param level: 显著性水平
            :return: 只有 (level) 的可能亏损大于 (return值), 取abs
            """
            return -(np.percentile(r, 100 * level)) * np.sqrt(period)

        if isinstance(self.array, pd.DataFrame):
            return self.array.aggregate(var_historic_sig, level=level, period=self.period)
        else:
            return var_historic_sig(self.array, level, self.period)

    @property
    def weighted_return(self, ):
        return self.weight.T @ self.mean_return

    def wr(self, weight):
        return weight.T @ self.mean_return

    @property
    def weighted_std(self):
        # print(self.weight.sum())
        # if self.weight.T @ self.cov @ self.weight <= 0:
        #     print(self.weight.sum())
        #     print(self.weight)
        #     print(self.cov)
        #     quit()

        return np.sqrt(self.weight.T @ self.cov @ self.weight)

    def ws(self, weight):
        return np.sqrt(weight.T @ self.cov @ weight)

    @property
    def sharpe(self):
        return (self.weighted_return - self.risk_free) / self.weighted_std


class Portfolio(Farray):
    def __init__(self, array, risk_free=0.02, period=252, ):
        # super(Portfolio, self).__init__()
        super().__init__(array, risk_free, period, )
        self.array: pd.DataFrame = array
        self.risk_free = risk_free
        self.period = period
        ...

    def seq_return(self):
        return self.array @ self.weight


    def indexer(self,std_, ret_, sharpe_):
        # mode: 0->min_std, 1->max_ret, 2->max_sharpe
        std_index = np.array(std_).argmin()
        ret_index = np.array(ret_).argmax()
        sharpe_index = np.array(sharpe_).argmax()
        return std_index, ret_index, sharpe_index


    def scatter_ret_vol(self, iters=1000, weight_inplace: bool = True):
        """
        Mont Carlo, use random weights combination to find the max sharpe
        :param iters:
        :param weight_inplace: whether use the max sharpe to inplace the self.woight
        :return:
        """
        ret_ = []
        std_ = []
        sharpe_ = []
        weight_ = []
        for _ in trange(iters):
            self.len = self.array.shape[1]
            self.weight: np.ndarray = np.random.random(self.len)
            self.weight /= self.weight.sum()
            self.weight.dtype = np.float128
            ret_.append(self.weighted_return)
            std_.append(self.weighted_std)
            sharpe_.append(self.sharpe)
            weight_.append(self.weight)
        # plt.scatter(std_, ret_, s=2, alpha=0.5, c=sharpe_, cmap='PuBu')
        plt.scatter(std_, ret_, s=2, alpha=0.5)
        # if risk_free is too high, sharpe does not fit well
        sharpe_index = sharpe_.index(max(sharpe_))

        plt.scatter(min(std_), ret_[std_.index(min(std_))], label='min std')
        plt.scatter(std_[ret_.index(max(ret_))], max(ret_), label='max return')
        plt.scatter(std_[sharpe_index], ret_[sharpe_index], label='max sharpe')

        if weight_inplace:
            self.weight = weight_[sharpe_index]

        plt.title('std_ret')
        plt.xlabel('std')
        plt.ylabel('return')
        allo_df = pd.DataFrame({'col': self.array.columns, 'weight': self.weight})
        print(f'Max Sharpe: {max(sharpe_)}', '\n', allo_df.T)

        plt.legend()
        plt.show()

    def var_historic(self, level=0.05):
        return -(np.percentile(self.array @ self.weight, 100 * level)) * np.sqrt(self.period)

    def scipy_opt(self, r):
        from scipy.optimize import minimize
        init_weight = np.ones((1, self.len)).flatten() / self.len
        weights_bounds = ((0, 1),) * self.len

        weights_sum_cons = {'type': 'eq', 'fun': lambda x: x.sum() - 1}

        return_target = {'type': 'eq', 'fun': lambda weights: self.wr(weights) - r}

        weights = minimize(self.ws, init_weight, method='SLSQP',
                           bounds=weights_bounds,
                           constraints=(weights_sum_cons, return_target))
        return weights.x

    def min_std_lim_math(self, optim=True, only_upper=False, title='', sharpe_color=True, weight_inplace_mode=None, num_points=100):
        """

        :param optim: use stats / use Analytical solution
        :param only_upper:
        :param title:
        :param sharpe_color: use sharpe value as cmap
        :param weight_inplace_mode: 0 for min_std, 1 for max_ret, 2 for max_sharpe
        :return:
        """
        ## This method is not stable, since huge number would emerge

        # lim: R^*, \Sigma w =1
        # L(w_1~w_n) =
        def std_lim(r):

            size = self.len
            mat = np.zeros((size + 2, size + 2))

            mat[:size, :size] = self.cov
            # row
            mat[size:size + 1, :size] = np.ones((1, size))
            mat[size + 1:size + 2, :size] = np.array(self.mean_return)
            # col
            mat[:size, size:size + 1] = -np.ones((size, 1))
            mat[:size, size + 1:size + 2] = -np.array(self.mean_return).reshape(-1, 1)

            # print('$$$$')
            # print(mat)
            # quit()
            b = np.zeros(size + 2)
            b[-2] = 1
            b[-1] = r
            line = np.linalg.inv(mat) @ b
            # print(line)
            return line

        ret_ = []
        std_ = []
        sharpe_ = []
        weights_ = []
        for r in tqdm.tqdm(np.linspace(min(self.mean_return), max(self.mean_return), num_points)):
            ret_.append(r)
            if not optim:
                self.weight = std_lim(r)[:self.len]
            else:
                self.weight = self.scipy_opt(r)

            # ws = self.weight.sum()
            sharpe_.append(self.sharpe)
            std_.append(self.weighted_std)
            weights_.append(self.weight)
        if only_upper:
            min_index = std_.index(min(std_))
            std_ = std_[min_index:]
            ret_ = ret_[min_index:]
            sharpe_ = sharpe_[min_index:]
        else:
            ...
        if sharpe_color:
            plt.scatter(std_, ret_, s=2, alpha=0.5, c=sharpe_)
        else:
            plt.scatter(std_, ret_, s=2, alpha=0.5, )

        if weight_inplace_mode:
            self.weight = weights_[self.indexer(std_, ret_, sharpe_)[weight_inplace_mode]]
        plt.xlabel('std')
        plt.ylabel('ret')
        plt.title(title)
        plt.show()

    def min_std_lim_opt(self):
        ...

    def CAPM(self, market_return):
        """

        :return:
        """
        # Capital Asset Pricing Modeling
        # E[p] = R_f + beta*(R_m - R_f) + alpha
        y = self.seq_return() - self.risk_free
        x = np.ones((self.date_len, 2))
        x[:, 0] = (market_return - self.risk_free)
        arr = np.linalg.inv(x.T @ x) @ x.T @ y
        alpha, beta = arr
        return alpha, beta


if __name__ == '__main__':
    a = np.array(pd.read_csv('temp.csv', index_col=0).iloc[:, 1])
    a = np.log(a + 1)
    far = Farray(a)
    far.draw_hist(quickshow=True, bins=1000)
    print(far.normal_check(level=0.05))
    far.qq()

