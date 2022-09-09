"""
基于多因子模型的归因分析
"""


import pandas as pd 


def cal_portfolio_exposure(port_holding, exposure):
    """
    给定组合权重与因子暴露，计算组合的因子暴露
    """
    port_holding = port_holding.stack().dropna()
    # 对齐index以便于加速运算
    exposure = pd.merge(port_holding.to_frame("_holding"), exposure,
                        on=['date', 'sid'], how='left').drop(['_holding'], axis=1)
    return exposure.mul(port_holding, axis=0).groupby(['date']).sum()


def decompose_portfolio_variance(port_holding, exposure, fcov, srisk, if_cr=True):
    """
    计算投资组合 ex ante的contribution to risk

    Ref: Qian (2005): On the Financial Interpretation of Risk Contribution

    逻辑过程:
         ``` latext
        \sigma^2 = w^T \Sigma w = (w^TB)\Sigma_f(B^T w) + w^TSw \\
        MCR = \dfrac{\partial{\sigma}}{\partial{w}} = \dfrac{B\Sigma_fB^Tw + Sw}{\sigma} \\
        CR = w_i * MCR_i\\
        \sum CR_i  = \sigma
        ```
    Parameters
    ---------
    port_holding: pd.Series
        投资组合权重
    trade_date: str
        YYYY-MM-DD 对应交易日
    if_cr: bool
        True: 计算Contribution to risk
        False: 计算 Marginal contribution to risk

    Return
    ------
    pd.DataFrame
        sid * (total_risk, specific_risk, systematic_risk)
    """
    port_holding = port_holding.reindex(exposure.index, fill_value=0.)

    # B\Sigma_fB^Tw
    sys_decomp = exposure.dot(fcov).dot(exposure.T).dot(port_holding)
    # Sw
    spec_decomp = (srisk ** 2).mul(port_holding, fill_value=0.)
    # Sw + B\Sigma_fB^Tw
    total_decomp = sys_decomp + spec_decomp 

    # 分别计算系统性部分的标准差，残差部分的标准差
    sys_std = port_holding.T.dot(sys_decomp) ** .5
    spec_std = port_holding.T.dot(spec_decomp) ** .5
    total_std = port_holding.T.dot(total_decomp) ** .5

    # MCR
    sys_decomp /= sys_std
    spec_decomp /= spec_std
    total_decomp /= total_std

    res = pd.DataFrame({'systematic_risk': sys_decomp,
                        'specific_risk': spec_decomp,
                        'total_risk': total_decomp})
    if if_cr:
        res['systematic_risk'] = res['systematic_risk'] * port_holding / sys_std
        res['specific_risk'] = res['specific_risk'] * port_holding / spec_std
        res['total_risk'] = res['total_risk'] * port_holding / total_std
        return res[res.sum(axis=1) > 0.]
