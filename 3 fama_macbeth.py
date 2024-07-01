import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Lasso
import warnings
warnings.filterwarnings("ignore")


# 温莎化
def winsorize(series, min_q=0.01, max_q=0.99):
    series = series.sort_values()
    max_range = series.quantile(max_q)
    min_range = series.quantile(min_q)
    return np.clip(series, min_range, max_range)


# 特征工程
def feature_engineering(dataset):
    for feature in dataset.columns[2:]:
        # 去极值处理
        dataset[feature] = winsorize(dataset[feature])

        if feature != 'ret':
            # 特征生成
            dataset[feature] = dataset.groupby('month')[feature].transform(lambda x: x.rank(pct=True))
            dataset[feature + '_pct'] = 1
            dataset[feature + '_pct'][(dataset[feature] >= 0.1) & (dataset[feature] < 0.2)] = 2
            dataset[feature + '_pct'][(dataset[feature] >= 0.2) & (dataset[feature] < 0.3)] = 3
            dataset[feature + '_pct'][(dataset[feature] >= 0.3) & (dataset[feature] < 0.4)] = 4
            dataset[feature + '_pct'][(dataset[feature] >= 0.4) & (dataset[feature] < 0.5)] = 5
            dataset[feature + '_pct'][(dataset[feature] >= 0.5) & (dataset[feature] < 0.6)] = 6
            dataset[feature + '_pct'][(dataset[feature] >= 0.6) & (dataset[feature] < 0.7)] = 7
            dataset[feature + '_pct'][(dataset[feature] >= 0.7) & (dataset[feature] < 0.8)] = 8
            dataset[feature + '_pct'][(dataset[feature] >= 0.8) & (dataset[feature] < 0.9)] = 9
            dataset[feature + '_pct'][dataset[feature] >= 0.9] = 10
    dataset = dataset.drop([f'R_{i}' for i in range(25)], axis=1)
    return dataset


# 训练模型，预测
def training_predicting(data):
    y_test_result_ks = pd.DataFrame(columns=['code', 'month', 'ret', 'prediction'])
    y_test_result_ls = pd.DataFrame(columns=['code', 'month', 'ret', 'prediction'])
    kitchen_sink_coefficients = pd.Series()
    lasso_coefficients = pd.Series()

    # 每月回归
    for month in pd.to_datetime(pd.date_range(start=pd.to_datetime('1997-1-1'),
                                              end=pd.to_datetime('2023-12-1'), freq='MS')):
        print(month)
        data_cross_reg = data[data['month'] == month]

        # 特征工程
        data_cross_reg = feature_engineering(data_cross_reg)

        # 划分被解释变量和解释变量
        x = data_cross_reg.drop(['code', 'month', 'ret'], axis=1)
        y = data_cross_reg['ret']
        x = sm.add_constant(x)

        # Kitchen Sink
        kitchen_sink = sm.OLS(y, x)
        kitchen_sink = kitchen_sink.fit()
        kitchen_sink_coefficients[month] = kitchen_sink.params

        # LASSO
        lasso = Lasso(alpha=0.1)
        lasso.fit(x, y)
        lasso_coefficients[month] = lasso.coef_

    # 用滑动窗口均值来预测
    ks_coefficients = pd.DataFrame(kitchen_sink_coefficients.tolist(), index=kitchen_sink_coefficients.index)
    ks_coefficients_mean = ks_coefficients.shift(1).rolling(window=60).mean().dropna(how='all')
    ks_coefficients_mean.columns = ['cons'] + [f'R_{i}' for i in range(25)]
    ls_coefficients = pd.DataFrame(lasso_coefficients.tolist(), index=lasso_coefficients.index)
    ls_coefficients_mean = ls_coefficients.shift(1).rolling(window=60).mean().dropna(how='all')
    ls_coefficients_mean.columns = ['cons'] + [f'R_{i}' for i in range(25)]
    for month in ks_coefficients_mean.index:
        data_predict = data[data['month'] == month]
        data_predict.insert(3, 'cons', 1)
        ks_predict = data_predict[['code', 'month', 'ret']].reset_index(drop=True)
        ls_predict = data_predict[['code', 'month', 'ret']].reset_index(drop=True)
        # Kitchen Sink
        ks_predict['prediction'] = data_predict.iloc[:, 3:].mul(ks_coefficients_mean.loc[month, :], axis=1).sum(axis=1).reset_index(drop=True)
        # LASSO
        ls_predict['prediction'] = data_predict.iloc[:, 3:].mul(ls_coefficients_mean.loc[month, :], axis=1).sum(axis=1).reset_index(drop=True)

        y_test_result_ks = pd.concat([ks_predict, y_test_result_ks])
        y_test_result_ls = pd.concat([ls_predict, y_test_result_ls])

    # 十分组
    y_test_result_ks['predict_rank'] = y_test_result_ks.groupby('month')['prediction'].transform(lambda x: x.rank(pct=True))
    y_test_result_ks['predict_pct'] = 1
    y_test_result_ks.loc[(y_test_result_ks['predict_rank'] >= 0.1) 
                         & (y_test_result_ks['predict_rank'] < 0.2), 'predict_pct'] = 2
    y_test_result_ks.loc[(y_test_result_ks['predict_rank'] >= 0.2) 
                         & (y_test_result_ks['predict_rank'] < 0.3), 'predict_pct'] = 3
    y_test_result_ks.loc[(y_test_result_ks['predict_rank'] >= 0.3) 
                         & (y_test_result_ks['predict_rank'] < 0.4), 'predict_pct'] = 4
    y_test_result_ks.loc[(y_test_result_ks['predict_rank'] >= 0.4) 
                         & (y_test_result_ks['predict_rank'] < 0.5), 'predict_pct'] = 5
    y_test_result_ks.loc[(y_test_result_ks['predict_rank'] >= 0.5) 
                         & (y_test_result_ks['predict_rank'] < 0.6), 'predict_pct'] = 6
    y_test_result_ks.loc[(y_test_result_ks['predict_rank'] >= 0.6) 
                         & (y_test_result_ks['predict_rank'] < 0.7), 'predict_pct'] = 7
    y_test_result_ks.loc[(y_test_result_ks['predict_rank'] >= 0.7) 
                         & (y_test_result_ks['predict_rank'] < 0.8), 'predict_pct'] = 8
    y_test_result_ks.loc[(y_test_result_ks['predict_rank'] >= 0.8) 
                         & (y_test_result_ks['predict_rank'] < 0.9), 'predict_pct'] = 9
    y_test_result_ks.loc[y_test_result_ks['predict_rank'] >= 0.9, 'predict_pct'] = 10
    
    y_test_result_ls['predict_rank'] = y_test_result_ls.groupby('month')['prediction'].transform(lambda x: x.rank(pct=True))
    y_test_result_ls['predict_pct'] = 1
    y_test_result_ls.loc[(y_test_result_ls['predict_rank'] >= 0.1) 
                         & (y_test_result_ls['predict_rank'] < 0.2), 'predict_pct'] = 2
    y_test_result_ls.loc[(y_test_result_ls['predict_rank'] >= 0.2) 
                         & (y_test_result_ls['predict_rank'] < 0.3), 'predict_pct'] = 3
    y_test_result_ls.loc[(y_test_result_ls['predict_rank'] >= 0.3) 
                         & (y_test_result_ls['predict_rank'] < 0.4), 'predict_pct'] = 4
    y_test_result_ls.loc[(y_test_result_ls['predict_rank'] >= 0.4) 
                         & (y_test_result_ls['predict_rank'] < 0.5), 'predict_pct'] = 5
    y_test_result_ls.loc[(y_test_result_ls['predict_rank'] >= 0.5) 
                         & (y_test_result_ls['predict_rank'] < 0.6), 'predict_pct'] = 6
    y_test_result_ls.loc[(y_test_result_ls['predict_rank'] >= 0.6) 
                         & (y_test_result_ls['predict_rank'] < 0.7), 'predict_pct'] = 7
    y_test_result_ls.loc[(y_test_result_ls['predict_rank'] >= 0.7) 
                         & (y_test_result_ls['predict_rank'] < 0.8), 'predict_pct'] = 8
    y_test_result_ls.loc[(y_test_result_ls['predict_rank'] >= 0.8) 
                         & (y_test_result_ls['predict_rank'] < 0.9), 'predict_pct'] = 9
    y_test_result_ls.loc[y_test_result_ls['predict_rank'] >= 0.9, 'predict_pct'] = 10

    y_test_result_ks.to_csv('./result/y_test_result_ks.csv')
    y_test_result_ls.to_csv('./result/y_test_result_ls.csv')
    return y_test_result_ks, y_test_result_ls


# 策略
def strategy(y_test_result):
    # 多空组合
    hedge_month = pd.DataFrame()
    long = y_test_result[y_test_result['predict_pct'] == 10].groupby('month')['ret'].mean().reset_index()
    short = y_test_result[y_test_result['predict_pct'] == 1].groupby('month')['ret'].mean().reset_index()
    hedge_month['ret'] = long['ret'] - short['ret']
    hedge_month['month'] = long['month']
    return hedge_month


def main():
    # 导入数据
    data = pd.read_csv('/model/data/data.csv', header=0, index_col=0)
    data['month'] = pd.to_datetime(data['month'])

    # 集中处理缺失值
    # 查看收益缺失值分布情况
    ret_na_num = data['ret'].isna().sum()
    # 若某只股票在第t月收益数据存在缺失（通常由股票连续停牌造成，共包含5282条缺失值），则剔除该股票在月份t上的所有数据
    data = data.dropna(subset='ret')
    # 查看特征缺失值分布情况
    na_num_any = data.isnull().any(axis=1).sum()
    na_num_col = data.iloc[:, 3:].apply(lambda x: x.isna().sum(), axis=0)
    # 因为特征是每月的前24个月的月度收益，所以特征的缺失来源于某月收益率的缺失，剔除所有含缺失值的行(共包含199258条缺失值)
    data = data.dropna()
    data = data.sort_values(['month', 'code']).reset_index(drop=True)

    # 训练模型，预测
    y_test_result_ks, y_test_result_ls = training_predicting(data)

    # 策略评估绩效
    # y_test_result_ks = pd.read_csv('./result/y_test_result_ks.csv')
    # y_test_result_ls = pd.read_csv('./result/y_test_result_ls.csv')
    hedge_month_ks = strategy(y_test_result_ks)
    hedge_month_ls = strategy(y_test_result_ls)
    hedge_month_ks.to_csv('./result/month_hedge_ret_ks.csv')
    hedge_month_ls.to_csv('./result/month_hedge_ret_ls.csv')


if __name__ == '__main__':
    main()
