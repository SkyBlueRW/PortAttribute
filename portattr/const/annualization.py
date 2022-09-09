"""
年化因子
"""


def annualization_factor(period):
    """
    返回对应周期(period)所需的年化因子

    Parameters
    ---------
    period: str [daily, weekly, monthly, yearly]
        定义调仓周期

    Returns
    -------
    annualization_factor : float
        年化因子
    """
    try:
        factor = ANNUALIZATION_FACTORS[period]
    except KeyError:
        raise ValueError(
                "Period应当为daily, weekly, monthly, yearly中的一种"
        )

    return factor


BDAYS_PER_YEAR = 244
BDAYS_PER_MONTH = 20
MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52
YEAR_PER_YEAR = 1

ANNUALIZATION_FACTORS = {
    'daily': BDAYS_PER_YEAR,
    'weekly': WEEKS_PER_YEAR,
    'monthly': MONTHS_PER_YEAR,
    'yearly': YEAR_PER_YEAR
}
