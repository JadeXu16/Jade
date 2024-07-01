import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('/Users/xuchuyu/PycharmProjects/Thesis/model/TRD_Mnth.csv', index_col=None, header=0)
data['month'] = pd.to_datetime(data['month'])


def percentile(series, min_q=0.05, max_q=0.95):
    series = series.sort_values()
    max_range = series.quantile(max_q)
    min_range = series.quantile(min_q)
    return np.clip(series, min_range, max_range)


market_ret = []
market_ret_mons = pd.Series()
for year in range(2001, 2024):
    print(year)
    data_test = data[data['month'].dt.year == year]

    # 缺失值填充：先进行时序中位数填充，再进行后向填充
    data_test['ret'] = data_test.groupby('code')['ret'].transform(lambda x: x.fillna(x.mean()))
    data_test['ret'] = data_test['ret'].fillna(method='bfill')
    # 去极值处理
    data_test['ret'] = percentile(data_test['ret'])

    # 缺失值填充：先进行时序中位数填充，再进行后向填充
    data_test['size'] = data_test.groupby('code')['size'].transform(lambda x: x.fillna(x.mean()))
    data_test['size'] = data_test['size'].fillna(method='bfill')
    # 去极值处理
    data_test['size'] = percentile(data_test['size'])

    Y_test = data_test[['code', 'month', 'ret', 'size']]
    Y_test = Y_test.sort_values('month')

    Y_test['weight'] = Y_test.groupby('month')['size'].transform(lambda x: x / x.sum()).reset_index(drop=True)
    Y_test['w_ret'] = Y_test['ret'] * Y_test['weight']
    market_ret_mon = Y_test.groupby('month')['w_ret'].sum()
    market_ret_mons = pd.concat([market_ret_mons, market_ret_mon])
    market_ret.append(market_ret_mon.sum())

pd.Series(market_ret).to_csv('./result/market_ret.csv')
market_ret_mons.to_csv('./result/market_ret_mon.csv')
