"""
收益率相关指标
"""

import numpy as np 
import pandas as pd 

from .const.annualization import annualization_factor 


def cal_ret_summary(ret, period, rf=0., benchmark=None):
    """
    计算多个收益率metrics

    Parameters
    ----------
    ret : pd.Series
        收益率。
    period: str 
        收益率周期 yearly, monthly, weekly, daily
    rt : float, default 0
        目标年化收益率。
    benchmark: pd.Series
        业绩基准
    
    Returns
    ------
    pd.DataFrame 指标
    """
    metric_name = []
    metric_value = []

    # 指标名称
    metric_name.append('sharpe_ratio')
    metric_name.append('sharpe-omega')
    metric_name.append('annual_return')
    metric_name.append('annual_std')
    metric_name.append('annual_downside_std')
    metric_name.append('max_drawdown')
    metric_name.append("max_drawdown_start")
    metric_name.append("max_darwdown_end")
    metric_name.append("long_drawdown_start")
    metric_name.append("long_darwdown_end")
    metric_name.append("win_rate")
    metric_name.append("pnl_ratio")
    metric_name.append("skewness")
    metric_name.append("kurtosis")

    # 策略表现
    metric_value.append('{0:.2f}'.format(cal_sharpe_ratio(ret, period, rf)))
    metric_value.append('{0:.2f}'.format(cal_sharpe_omega(ret, rf)))
    metric_value.append('{0:.1%}'.format(cal_annual_ret(ret, period)))
    metric_value.append('{0:.1%}'.format(cal_annual_std(ret, period)))
    metric_value.append('{0:.1%}'.format(cal_annual_downside_std(ret, period, rf)))
    metric_value.append('{0:.1%}'.format(cal_daily_drawdown(ret).min()))
    metric_value.append(cal_max_drawdown_start(ret))
    metric_value.append(cal_max_drawdown_end(ret))
    dd_s, dd_e = cal_long_drawdown_recovery_start_end(ret)
    metric_value.append(dd_s)
    metric_value.append(dd_e)
    metric_value.append('{0:.1%}'.format(cal_win_rate(ret)))
    metric_value.append('{0:.1%}'.format(cal_pnl_ratio(ret)))
    metric_value.append('{0:.2f}'.format(ret.dropna().skew()))
    metric_value.append('{0:.2f}'.format(ret.dropna().kurtosis()))

    # 超额收益
    if benchmark is not None:
        active_value = []
        benchmark_value = []
        active_rtn = ret - benchmark
        active_value.append('{0:.2f}'.format(cal_sharpe_ratio(active_rtn, period, rf)))
        active_value.append('{0:.2f}'.format(cal_sharpe_omega(active_rtn, rf)))
        active_value.append('{0:.1%}'.format(cal_annual_ret(active_rtn, period)))
        active_value.append('{0:.1%}'.format(cal_annual_std(active_rtn, period)))
        active_value.append('{0:.1%}'.format(cal_annual_downside_std(active_rtn, period, rf)))
        active_value.append('{0:.1%}'.format(cal_daily_drawdown(active_rtn).min()))
        active_value.append(cal_max_drawdown_start(active_rtn))
        active_value.append(cal_max_drawdown_end(active_rtn))
        dd_s, dd_e = cal_long_drawdown_recovery_start_end(active_rtn)
        active_value.append(dd_s)
        active_value.append(dd_e)
        active_value.append('{0:.1%}'.format(cal_win_rate(active_rtn)))
        active_value.append('{0:.1%}'.format(cal_pnl_ratio(active_rtn)))
        active_value.append('{0:.2f}'.format(active_rtn.dropna().skew()))
        active_value.append('{0:.2f}'.format(active_rtn.dropna().kurtosis()))


        benchmark_value.append('{0:.2f}'.format(cal_sharpe_ratio(benchmark, period, rf)))
        benchmark_value.append('{0:.2f}'.format(cal_sharpe_omega(benchmark, rf)))
        benchmark_value.append('{0:.1%}'.format(cal_annual_ret(benchmark, period)))
        benchmark_value.append('{0:.1%}'.format(cal_annual_std(benchmark, period)))
        benchmark_value.append('{0:.1%}'.format(cal_annual_downside_std(benchmark, period, rf)))
        benchmark_value.append('{0:.1%}'.format(cal_daily_drawdown(benchmark).min()))
        benchmark_value.append(cal_max_drawdown_start(benchmark))
        benchmark_value.append(cal_max_drawdown_end(benchmark))
        dd_s, dd_e = cal_long_drawdown_recovery_start_end(benchmark)
        benchmark_value.append(dd_s)
        benchmark_value.append(dd_e)
        benchmark_value.append('{0:.1%}'.format(cal_win_rate(benchmark)))
        benchmark_value.append('{0:.1%}'.format(cal_pnl_ratio(benchmark)))
        benchmark_value.append('{0:.2f}'.format(benchmark.dropna().skew()))
        benchmark_value.append('{0:.2f}'.format(benchmark.dropna().kurtosis()))


        return pd.DataFrame({
            'Strategy': metric_value,
            'Benchmark': benchmark_value,
            'Active': active_value
        }, index=metric_name)
    else:
        return pd.Series(metric_value, index=metric_name)


def cal_yearly_ret_summary(ret, period, rf=0.):
    """
    逐年计算多个收益率指标

    Parameters
    ----------
    ret : pd.Series
        对数收益率。
    period: str 
        收益率周期 yearly, monthly, weekly, daily
    rt : float, default 0
        目标年化收益率。
    benchmark: pd.Series
        业绩基准
    
    Returns
    ------
    pd.DataFrame 指标
        metric * year
    """
    return ret.groupby(lambda x: x.year).apply(cal_ret_summary, period, rf).unstack().T





def cal_ret(nav):
    """
    给定净值计算几何收益率

    Parameters
    ----------
    nav: pd.Series
        策略净值
    
    Returns
    -------
    pd.Series
        收益率
    """
    return nav.ffill().pct_change()


def cal_sharpe_omega(ret, L_threshold=0.):
    """
    计算Sharpe-Omega指标

    Parameters
    ----------
    ret: pd.Series
        收益率
    L_threshold: float
        收益阈值
    
    Returns
    -------
    float
    """
    if ret.size:
        res = L_threshold - ret 
        res = np.sum(res[res >=0]) / len(ret)
        res = (ret.mean() - L_threshold) / res 
        return res 
    else:
        return np.nan


def cal_annual_ret(ret, period):
    """
    计算年化收益率

    Parameters
    ---------
    ret: pd.Series
        收益率
    period: str 
        收益率周期 yearly, monthly, weekly, daily
    
    Returns
    --------
    float
        年化收益率
    """
    if ret.size:
        return np.prod(1+ret) ** (annualization_factor(period) / len(ret)) - 1.
    else:
        return np.nan


def cal_annual_std(ret, period):
    """
    计算年化波动率。

    Parameters
    ----------
    ret : pd.Series
        收益率。
    period: str 
        收益率周期 yearly, monthly, weekly, daily

    Returns
    -------
    float
        年化波动率。
    """
    if ret.size > 1:
        return ret.std() * annualization_factor(period) ** 0.5
    else:
        return np.nan


def cal_annual_downside_std(ret, period, rt=0.):
    """
    计算年化下行波动率。

    Reference: https://www.investopedia.com/terms/s/semivariance.asp

    Parameters
    ----------
    ret : pd.Series
        收益率。
    period: str 
        收益率周期 yearly, monthly, weekly, daily
    rt : float, default 0
        目标年化收益率。

    Returns
    -------
    float
        年化下行波动率
    """
    if ret.size > 1:
        ret = ret.copy()
        ret = ret - rt
        return ((ret[ret > 0] ** 2).sum() / len(ret) * annualization_factor(period)) ** 0.5
    else:
        return np.nan


def cal_daily_drawdown(ret):
    """
    计算每日累计回撤。

    Parameters
    ----------
    ret : pd.Series
        收益率。

    Returns
    -------
    pd.Series
        每日回撤。
    """
    cum_ret = ret.add(1.).cumprod()
    max_so_far = cum_ret.expanding(min_periods=1).max()
    return (cum_ret - max_so_far) / max_so_far


def cal_max_drawdown_end(ret):
    """
    计算最大回撤结束日期

    Parameters
    ----------
    ret : pd.Series
        对数收益率。

    Returns
    -------
    datetime
    """
    daily_drawdown = cal_daily_drawdown(ret)
    return pd.to_datetime(daily_drawdown.idxmin()).strftime("%Y-%m-%d")


def cal_max_drawdown_start(ret):
    """
    计算最大回撤开始日期

    Parameters
    ----------
    ret : pd.Series
        收益率。

    Returns
    -------
    datetime
    """
    daily_drawdown = cal_daily_drawdown(ret)
    end_date = daily_drawdown.idxmin()
    return pd.to_datetime(ret.cumsum().loc[: end_date].idxmax()).strftime("%Y-%m-%d")


def cal_long_drawdown_recovery_start_end(ret):
    """
    计算最长回撤开始与结束日期

    Parameters
    ---------
    ret: pd.Series
        收益率
    
    Returns
    -------
    start, end，最长回撤的开始与结束
    """
    ret = ret.fillna(0.)
    arr = ret.add(1.).cumprod().values
    max_seen = arr[0]
    ddd_start, ddd_end = 0, 0
    ddd = 0
    start = 0
    in_draw_down = False

    for i in range(len(arr)):
        if arr[i] > max_seen:
            if in_draw_down:
                in_draw_down = False
                if i - start > ddd:
                    ddd = i - start
                    ddd_start = start
                    ddd_end = i - 1
            max_seen = arr[i]
        elif arr[i] < max_seen:
            if not in_draw_down:
                in_draw_down = True
                start = i - 1

    if arr[i] < max_seen:
        if i - start > ddd:
            return pd.Timestamp(ret.index[start]).strftime("%Y-%m-%d"), pd.Timestamp(ret.index[i]).strftime("%Y-%m-%d")

    return pd.Timestamp(str(ret.index[ddd_start])).strftime("%Y-%m-%d"), pd.Timestamp(str(ret.index[ddd_end])).strftime("%Y-%m-%d")




def cal_sharpe_ratio(ret, period, rf=0.):
    """
    计算年化夏普比率。

    Parameters
    ----------
    ret : pd.Series
        收益率。
    period: str 
        收益率周期 yearly, monthly, weekly, daily
    rf : float
        年化无风险利率。

    Returns
    -------
    float
        年化夏普比率。
    """
    vol = cal_annual_std(ret, period)
    if vol > 0:
        return (cal_annual_ret(ret, period) - rf) / vol
    else:
        return float('nan')


def cal_win_rate(ret):
    """计算日收益胜率。

    Parameters
    ----------
    ret : pd.Series
        收益率。

    Returns
    -------
    float
        日收益胜率。
    """
    if ret.size:
        return (ret >= 0).sum() / ret.size
    else:
        return float('nan')


def cal_pnl_ratio(ret):
    """计算日收益盈亏比。

    Parameters
    ----------
    ret : pd.Series
        收益率。

    Returns
    -------
    float
        日收益盈亏比。
    """
    return -ret[ret >= 0].mean() / ret[ret < 0].mean()