import pandas as pd
import numpy as np
#http://statistica.ru/local-portals/quality-control/indeksy-prigodnosti-protsessa/
def calc_stats(s):
    #Up и Lp – 99.865 и 0.135 процентили соответственно
    Up = s.quantile(q = 0.99865)
    Lp = s.quantile(q = 0.00135)
    M = s.median()
    return Up, Lp, M

def calc_pp(s, NGD, VGD):
    Up, Lp, _  = calc_stats(s)
    #Потенциальная пригодность (Cp)
    Pp = (VGD-NGD)/(Up-Lp) 
    return Pp

def calc_ppu(s, NGD, VGD):
    Up, Lp, M  = calc_stats(s)
    Ppu = (VGD-M)/(Up - M) 
    #Подтвержденное качество (Cpk)

    return Ppu

def calc_ppl(s,NGD,VGD):
    Up, Lp, M = calc_stats(s)
    Ppl = (M - NGD)/(M - Lp) 
    return Ppl


def calc_ppk(s, NGD, VGD):
    return min((calc_ppl(s, NGD, VGD), calc_ppu(s, NGD, VGD)))

def suggest_control_limits(data: ( pd.Series, np.array), sigma_level: float = 3.0):
    """
    Given a data set and a sigma level, will return a dict containing the `upper_control_limit` and \
    `lower_control_limit`. values

    :param data: the data to be analyzed
    :param sigma_level: the sigma level; the default value is 3.0, but some users \
    may prefer a higher sigma level for their process
    :return: a `dict` containing the `upper_control_limit` and `lower_control_limit` keys
    """

    median = data.median()
    sigma = data.std()

    return median - sigma_level * sigma, median + sigma_level * sigma

def control_beyond_limits(data: ( pd.Series, np.array),
                          upper_control_limit: (int, float), lower_control_limit: (int, float)):
    """
    Returns a pandas.Series with all points which are beyond the limits.

    :param data: The data to be analyzed
    :param upper_control_limit: the upper control limit
    :param lower_control_limit: the lower control limit
    :return: a pandas.Series object with all out-of-control points
    """

    return data.where((data > upper_control_limit) | (data < lower_control_limit)).dropna()


def control_zone_a(data: (pd.Series, np.array),
                   upper_control_limit: (int, float), lower_control_limit: (int, float)):
    """
    Returns a pandas.Series containing the data in which 2 out of 3 are in zone A or beyond.

    :param data: The data to be analyzed
    :param upper_control_limit: the upper control limit
    :param lower_control_limit: the lower control limit
    :return: a pandas.Series object with all out-of-control points
    """

    spec_range = (upper_control_limit - lower_control_limit) / 2
    spec_center = lower_control_limit + spec_range
    zone_b_upper_limit = spec_center + 2 * spec_range / 3
    zone_b_lower_limit = spec_center - 2 * spec_range / 3

    # looking for violations in which 2 out of 3 are in zone A or beyond
    violations = []
    for i in range(len(data) - 2):
        points = data[i:i+3].to_numpy()

        count = 0
        for point in points:
            if point < zone_b_lower_limit or point > zone_b_upper_limit:
                count += 1

        if count >= 2:
            index = i + np.arange(3)
            violations.append(pd.Series(data=points, index=index))


    if len(violations) == 0:
        return pd.Series()

    s = pd.concat(violations)
    s = s.loc[~s.index.duplicated()]
    return s


def control_zone_b(data: ( pd.Series, np.array),
                   upper_control_limit: (int, float), lower_control_limit: (int, float)):
    """
    Returns a pandas.Series containing the data in which 4 out of 5 are in zone B or beyond.

    :param data: The data to be analyzed
    :param upper_control_limit: the upper control limit
    :param lower_control_limit: the lower control limit
    :return: a pandas.Series object with all out-of-control points
    """


    spec_range = (upper_control_limit - lower_control_limit) / 2
    spec_center = lower_control_limit + spec_range
    zone_c_upper_limit = spec_center + spec_range / 3
    zone_c_lower_limit = spec_center - spec_range / 3

    # looking for violations in which 2 out of 3 are in zone A or beyond
    violations = []
    for i in range(len(data) - 5):
        points = data[i:i+5].to_numpy()

        count = 0
        for point in points:
            if point < zone_c_lower_limit or point > zone_c_upper_limit:
                count += 1

        if count >= 4:
            index = i + np.arange(5)
            violations.append(pd.Series(data=points, index=index))


    if len(violations) == 0:
        return pd.Series()

    s = pd.concat(violations)
    s = s.loc[~s.index.duplicated()]
    return s


def control_zone_c(data: ( pd.Series, np.array),
                   upper_control_limit: (int, float), lower_control_limit: (int, float)):
    """
    Returns a pandas.Series containing the data in which 7 consecutive points are on the same side.

    :param data: The data to be analyzed
    :param upper_control_limit: the upper control limit
    :param lower_control_limit: the lower control limit
    :return: a pandas.Series object with all out-of-control points
    """

    spec_range = (upper_control_limit - lower_control_limit) / 2
    spec_center = lower_control_limit + spec_range

    # looking for violations in which 2 out of 3 are in zone A or beyond
    violations = []
    for i in range(len(data) - 6):
        points = data[i:i+7].to_numpy()

        count = 1
        above = data[i] > spec_center
        for point in points[1:]:
            if above is True:
                if point > spec_center:
                    count += 1
                else:
                    break
            else:
                if point < spec_center:
                    count += 1
                else:
                    break

        if count >= 7:
            index = i + np.arange(7)
            violations.append(pd.Series(data=points, index=index))

    if len(violations) == 0:
        return pd.Series()

    s = pd.concat(violations)
    s = s.loc[~s.index.duplicated()]
    return s


def control_zone_trend(data: (pd.Series, np.array)):
    """
    Returns a pandas.Series containing the data in which 7 consecutive points trending up or down.

    :param data: The data to be analyzed
    :return: a pandas.Series object with all out-of-control points
    """


    # looking for violations in which 2 out of 3 are in zone A or beyond
    violations = []
    for i in range(len(data) - 6):
        points = data[i:i+7].to_numpy()

        up = 0
        down = 0
        for j in range(1, 7):
            if points[j] > points[j-1]:
                up += 1
            elif points[j] < points[j-1]:
                down += 1

        if up >= 6 or down >= 6:
            index = i + np.arange(7)
            violations.append(pd.Series(data=points, index=index))

    if len(violations) == 0:
        return pd.Series()

    s = pd.concat(violations)
    s = s.loc[~s.index.duplicated()]
    return s


def control_zone_mixture(data: (pd.Series, np.array),
                         upper_control_limit: (int, float), lower_control_limit: (int, float)):
    """
    Returns a pandas.Series containing the data in which 8 consecutive points occur with none in zone C

    :param data: The data to be analyzed
    :param upper_control_limit: the upper control limit
    :param lower_control_limit: the lower control limit
    :return: a pandas.Series object with all out-of-control points
    """

    spec_range = (upper_control_limit - lower_control_limit) / 2
    spec_center = lower_control_limit + spec_range
    zone_c_upper_limit = spec_center + spec_range / 3
    zone_c_lower_limit = spec_center - spec_range / 3

    # looking for violations in which 2 out of 3 are in zone A or beyond
    violations = []
    for i in range(len(data) - 7):
        points = data[i:i+8].to_numpy()

        count = 0
        for point in points:
            if not zone_c_lower_limit < point < zone_c_upper_limit:
                count += 1
            else:
                break

        if count >= 8:
            index = i + np.arange(8)
            violations.append(pd.Series(data=points, index=index))

    if len(violations) == 0:
        return pd.Series()

    s = pd.concat(violations)
    s = s.loc[~s.index.duplicated()]
    return s


def control_zone_stratification(data: ( pd.Series, np.array),
                                upper_control_limit: (int, float), lower_control_limit: (int, float)):
    """
    Returns a pandas.Series containing the data in which 15 consecutive points occur within zone C

    :param data: The data to be analyzed
    :param upper_control_limit: the upper control limit
    :param lower_control_limit: the lower control limit
    :return: a pandas.Series object with all out-of-control points
    """

    spec_range = (upper_control_limit - lower_control_limit) / 2
    spec_center = lower_control_limit + spec_range
    zone_c_upper_limit = spec_center + spec_range / 3
    zone_c_lower_limit = spec_center - spec_range / 3

    # looking for violations in which 2 out of 3 are in zone A or beyond
    violations = []
    for i in range(len(data) - 14):
        points = data[i:i+15].to_numpy()

        points = points[np.logical_and(points < zone_c_upper_limit, points > zone_c_lower_limit)]

        if len(points) >= 15:
            index = i + np.arange(15)
            violations.append(pd.Series(data=points, index=index))


    if len(violations) == 0:
        return pd.Series()

    s = pd.concat(violations)
    s = s.loc[~s.index.duplicated()]
    return s


def control_zone_overcontrol(data: ( pd.Series, np.array),
                             upper_control_limit: (int, float), lower_control_limit: (int, float)):
    """
    Returns a pandas.Series containing the data in which 14 consecutive points are alternating above/below the center.

    :param data: The data to be analyzed
    :param upper_control_limit: the upper control limit
    :param lower_control_limit: the lower control limit
    :return: a pandas.Series object with all out-of-control points
    """

    spec_range = (upper_control_limit - lower_control_limit) / 2
    spec_center = lower_control_limit + spec_range

    # looking for violations in which 2 out of 3 are in zone A or beyond
    violations = []
    for i in range(len(data) - 14):
        points = data[i:i+14].to_numpy()
        odds = points[::2]
        evens = points[1::2]

        if odds[0] > 0.0:
            odds = odds[odds > spec_center]
            evens = evens[evens < spec_center]
        else:
            odds = odds[odds < spec_center]
            evens = evens[evens > spec_center]

        if len(odds) == len(evens) == 7:
            index = i + np.arange(14)
            violations.append(pd.Series(data=points, index=index))


    if len(violations) == 0:
        return pd.Series()

    s = pd.concat(violations)
    s = s.loc[~s.index.duplicated()]
    return s

print('Hi')
