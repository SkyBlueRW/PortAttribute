"""
绘图方法
"""

import numpy as np 
import pandas as pd 
from scipy.stats import norm
import matplotlib 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib import colors
from IPython.display import display


def print_heatmap(df, format='{:.1%}'):
    """
    以heatmap的形式进行打印， 全df进行颜色比较
    """
    # color map
    cm_red = plt.cm.get_cmap(sns.light_palette("red", as_cmap=True))
    cm_green = plt.cm.get_cmap(sns.light_palette("green", as_cmap=True))
    # normalizer
    max_red = df.max().max()
    max_green = abs(df.min().min())
    normalize_red = colors.Normalize(0, max_red)
    normalize_green = colors.Normalize(0, max_green)
    # backgroud color
    def bg_color(ser):
        c = [colors.rgb2hex(cm_red(normalize_red(x))) if x > 0 else colors.rgb2hex(cm_green(normalize_green(-x))) for x in ser.values]
        return ['background-color: %s' % color for color in c]
    styled_df = df.style.apply(bg_color).format(format)
    display(styled_df)


def print_heatmap_1d(df, format="{:.1%}", axis=1):
    """
    以heatmap形式进行打印，以列或者行进行颜色比较
    """
    cm = sns.diverging_palette(150, 10, as_cmap=True)
    styled_df = df.fillna(0.).style.format(format).background_gradient(cmap=cm, axis=axis)
    display(styled_df)


def print_monthly_heatmap(srs, format='{:.1%}', func=np.nanmean):
    """
    """
    df = srs.to_frame('val')
    df['year'] = df.index.year
    df['month'] = df.index.month
    res = df.groupby(['year', 'month'])['val'].apply(lambda x: func(x)).unstack()

    print_heatmap(res, format)


def setup_chinese_font(font='FangSong'):
    """
    设置中文
    """
    rcparams = {
        'font.sans-serif': [font]
    }
    matplotlib.rcParams.update(rcparams)


def setup_plotting_context(font='FangSong'):
    """
    绘图参数设置
    """
    rcparams = {
        'legend.fontsize': 'large',
        'figure.figsize': (8, 6),
        'savefig.dpi': 90,
        'font.sans-serif': [font],
        'axes.unicode_minus': False,
        'axes.labelsize': 'x-large',
        'axes.titlesize': 'x-large',
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large',
        'figure.facecolor': '#F6F6F6',
        'axes.facecolor': '#F6F6F6',
        'axes.edgecolor': '#D3D3D3',
        'text.color': '#555555',
        'grid.color': '#B1B1B1'
    }

    matplotlib.rcParams.update(rcparams)


def plot_nav_ts(nav, ax=None, title="", benchmark=None):
    """
    绘制策略收益的走势以及回撤区间, 

    绘图的title是maxdd， 与dd_days

    Parameters
    ---------
    ret : pd.Series (index: date)
        策略每日收益
    ax: matplotlib.Axes
        绘图在上边的Axes
    Returns
    ------
    ax: matplotlib.Axes
    """
    
    # 回撤信息
    max_here = nav.expanding(min_periods=1).max()
    dd_here = nav / max_here - 1
    # 最大回撤的开始与结束
    tmp = dd_here.sort_values().head(1)
    max_dd = round(float(tmp.values), 3)
    end_date = tmp.index.strftime('%Y-%m-%d')[0]
    tmp = nav[:end_date]
    tmp = tmp.sort_values(ascending=False).head(1)
    start_date = tmp.index.strftime('%Y-%m-%d')[0]
    dt_range = len(pd.period_range(start_date, end_date))

    # 开始绘图
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))
    
    ax.get_xaxis().set_minor_locator(
        matplotlib.ticker.AutoMinorLocator()
    )
    ax.get_yaxis().set_minor_locator(
        matplotlib.ticker.AutoMinorLocator()
    )

    ax.plot(nav-1., label=title, alpha=1, linewidth=1.5, color='#aa4643')    
    plt.title("{} || maxdd:{}  dd_days:{}({}~{})".format(title, max_dd, dt_range, start_date, end_date))
    md_start = pd.to_datetime(start_date)
    md_end = pd.to_datetime(end_date)
    ax.axvspan(md_start, md_end, alpha=0.2, color='#00FF00')
    ax.axhline(y=0., color='black', linewidth=1.5)

    if benchmark is not None:
        ax.plot(benchmark-1., color='Blue', alpha=1, linewidth=1.5)

    return ax 


def plot_heatmap(df, ax=None, title=None, annot=True):
    """
    绘制热力图
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(12, 12))
    sns.heatmap(
            df,
            annot=annot,
            alpha=1.0,
            center=0.0,
            annot_kws={'size': 10},
            linewidths=0.01,
            linecolor='white',
            cmap=sns.diverging_palette(150, 10, as_cmap=True),
            cbar=False,
            ax=ax
        )
    ax.set(ylabel="", xlabel="")
    if title is None:
        title = ""
    ax.set_title(title)
    plt.yticks(rotation=0)

    return ax


def plot_empirical_distribution(srs, ax=None, title=None):
    """
    画出经验分布
    """
    mean, median, std = round(srs.mean(), 4), round(srs.median(), 4), round(srs.std(), 4)
    skew = round(srs.skew(), 4)
    kurt = round(srs.kurtosis(), 4)
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(14, 6))

    # histogram
    n, bins, patches = plt.hist(srs.values, 100, density=1, facecolor='green', alpha=0.5)
    # best fit line
    # y = matplotlib.mlab.normpdf(bins, mean, std)
    y = norm.pdf(bins, mean, std)
    plt.plot(bins, y, 'r--')
    if title is None:
        title = "Empirical Distribution"
    ax.set_title("{3}: Mean={0}, Median={1}, std={2}, skew={4}, kurt={5}".format(mean, median, std, title, skew, kurt))

    return ax


def plot_nav_summary(nav, benchmark=None):
    """
    """
    pass 