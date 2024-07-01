import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import itertools
import warnings
warnings.filterwarnings("ignore")

num = ''
output_url = './result' + num + '/'


# 集中处理缺失值并统计数据分布
def data_process(data):
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
    # data['year'] = data['month'].dt.year
    # data = data[data['year'] != 2024]
    #
    # # 统计数据分布
    # data_num = data.groupby('month').size().reset_index()
    #
    # plt.figure(figsize=(15, 5))
    # plt.bar(data_num['month'], data_num[0], width=10, color='black', alpha=0.7)

    return data


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


# 调参：网格搜索，交叉验证
def adj_parameter(x_train, y_train):
    # 网格搜索
    param_grid = {
        'n_estimators': [200],
        'max_features': [2],
        'max_depth': [2, 3]
    }
    rf = RandomForestRegressor()

    # 交叉验证
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=4)
    grid_search.fit(x_train, y_train)

    best_params = grid_search.best_params_
    print(best_params)
    return best_params


# variable importance
def variable_importance(rf_regressor, year, var_importance_s):
    var_importance = pd.Series(rf_regressor.feature_importances_, index=[f'R_{i}' for i in range(25)]).rank()
    var_importance_s.loc[year] = var_importance.T
    return var_importance_s


# partial derivatives
def partial_derivatives(y_predict_pd, x_test, year, rf_regressor):
    x_test_pd = x_test.copy()
    combinations = list(itertools.combinations(x_test.columns[[9, 22, 24, 21]], 2))
    for r1, r2 in combinations:
        for d1 in range(1, 11):
            for d2 in range(1, 11):
                x_test_pd[r1] = d1
                x_test_pd[r2] = d2
                y_predict_pd_0 = pd.Series({'year': year,
                                            'r1': r1,
                                            'd1': d1,
                                            'r2': r2,
                                            'd2': d2,
                                            'ret': rf_regressor.predict(x_test_pd).mean()})
                y_predict_pd = pd.concat([y_predict_pd,  y_predict_pd_0.to_frame().T], axis=0, ignore_index=True)
    return y_predict_pd


# 训练模型，预测
def training_predicting(data):
    y_test_result = pd.DataFrame(columns=['code', 'month', 'ret', 'prediction', 'year'])
    var_importance_s = pd.DataFrame(index=[i for i in range(2002, 2024)], columns=[f'R_{i}' for i in range(25)])
    y_predict_pd = pd.DataFrame(columns=['year', 'r1', 'd1', 'r2', 'd2', 'ret'])
    best_parameters = 0

    # 每年建模并评估
    for year in range(2002, 2024):
        print(year)
        if year == 2002:
            data_train = data[data['month'].dt.year < year]
            data_test = data[data['month'].dt.year == year]
        else:
            data_train = data[(data['month'].dt.year >= year - 15) & (data['month'].dt.year < year)]
            data_test = data[data['month'].dt.year == year]

        # 训练集特征工程
        data_train = feature_engineering(data_train)
        # 测试集特征工程
        data_test = feature_engineering(data_test)

        # 统计训练和测试样本数
        x_train = data_train.drop(['code', 'month', 'ret'], axis=1)
        y_train = data_train['ret']
        print(f'训练集样本数：{len(x_train)}')
        x_test = data_test.drop(['code', 'month', 'ret'], axis=1)
        y_test = data_test[['code', 'month', 'ret']]
        print(f'测试集样本数：{len(x_test)}')
        print(f'训练集和测试集样比例：{len(x_train) / len(x_test)}')

        # 训练模型
        best_parameters = adj_parameter(x_train, y_train)
        rf_regressor = RandomForestRegressor(**best_parameters, criterion='squared_error', random_state=16)
        rf_regressor.fit(x_train, y_train)
        y_test['prediction'] = rf_regressor.predict(x_test)
        y_test['year'] = year

        # 计算训练集mse
        mse_in = mean_squared_error(y_train, rf_regressor.predict(x_train))
        print(f'mse_in: {mse_in}')
        # 计算测试集mse
        mse_out = mean_squared_error(y_test['ret'], y_test['prediction'])
        print(f'mse_ot: {mse_out}')
        print(f'样本外和样本内mse比例：{mse_out / mse_in}')
        y_test_result = pd.concat([y_test_result, y_test])

        # # variable importance
        # var_importance_s = variable_importance(rf_regressor, year, var_importance_s)

        # partial derivatives
        y_predict_pd = partial_derivatives(y_predict_pd, x_test, year, rf_regressor)

    # 保存结果
    # # var imp
    # var_importance_s.to_csv(output_url + 'var_imp.csv')
    # var_importance_median = var_importance_s.median()
    # var_importance_upper = var_importance_s.quantile(0.25)
    # var_importance_lower = var_importance_s.quantile(0.75)
    # var_importance_quartile = pd.DataFrame({'upper': var_importance_upper,
    #                                         'median': var_importance_median,
    #                                         'lower': var_importance_lower})
    # var_importance_quartile.to_csv(output_url + 'var_imp_quartile.csv')
    # partial derivatives
    y_predict_pd.to_csv(output_url + 'pd_2.csv')

    # 十分组
    y_test_result['predict_rank'] = y_test_result.groupby('month')['prediction'].transform(lambda x: x.rank(pct=True))
    y_test_result['predict_pct'] = 1
    y_test_result.loc[(y_test_result['predict_rank'] >= 0.1) & (y_test_result['predict_rank'] < 0.2), 'predict_pct'] = 2
    y_test_result.loc[(y_test_result['predict_rank'] >= 0.2) & (y_test_result['predict_rank'] < 0.3), 'predict_pct'] = 3
    y_test_result.loc[(y_test_result['predict_rank'] >= 0.3) & (y_test_result['predict_rank'] < 0.4), 'predict_pct'] = 4
    y_test_result.loc[(y_test_result['predict_rank'] >= 0.4) & (y_test_result['predict_rank'] < 0.5), 'predict_pct'] = 5
    y_test_result.loc[(y_test_result['predict_rank'] >= 0.5) & (y_test_result['predict_rank'] < 0.6), 'predict_pct'] = 6
    y_test_result.loc[(y_test_result['predict_rank'] >= 0.6) & (y_test_result['predict_rank'] < 0.7), 'predict_pct'] = 7
    y_test_result.loc[(y_test_result['predict_rank'] >= 0.7) & (y_test_result['predict_rank'] < 0.8), 'predict_pct'] = 8
    y_test_result.loc[(y_test_result['predict_rank'] >= 0.8) & (y_test_result['predict_rank'] < 0.9), 'predict_pct'] = 9
    y_test_result.loc[y_test_result['predict_rank'] >= 0.9, 'predict_pct'] = 10

    y_test_result.to_csv(output_url + 'y_test_result.csv')
    return y_test_result


# 策略
def strategy(y_test_result):
    # 多空组合
    long = y_test_result[y_test_result['predict_pct'] == 10].groupby(['year', 'month'])['ret'].mean().reset_index()
    short = y_test_result[y_test_result['predict_pct'] == 1].groupby(['year', 'month'])['ret'].mean().reset_index()
    long_year = long.groupby('year')['ret'].sum()
    short_year = short.groupby('year')['ret'].sum()
    hedge_year = long_year - short_year
    long_month = long.groupby('month')['ret'].sum()
    short_month = short.groupby('month')['ret'].sum()
    hedge_month = long_month - short_month
    hedge_month.to_csv(output_url + 'month_hedge_ret.csv')

    # 市场
    market_year = pd.read_csv('result/market_ret.csv', header=0, index_col=0)
    market_year = pd.Series(market_year['market'])

    # 输出结果
    all_ret = pd.DataFrame({'long': long_year,
                            'short': short_year,
                            'hedge': hedge_year,
                            'market': market_year})
    all_ret.to_csv(output_url + 'all_ret.csv')

    # 净值
    # long
    long_val = y_test_result[y_test_result['predict_pct'] == 10].groupby('month')['ret'].mean()
    long_val += 1
    long_val = long_val.cumprod()
    # short
    short_val = y_test_result[y_test_result['predict_pct'] == 1].groupby('month')['ret'].mean()
    short_val += 1
    short_val = short_val.cumprod()
    # market
    market = pd.read_csv('result/market_ret_mon.csv',
                         index_col=None, header=0)
    market_val = market.groupby('month').mean()
    market_val += 1
    market_val = market_val.cumprod()
    market_val = pd.Series(market_val['market'])
    # risk—free
    risk_free = pd.read_csv('data/risk_free_rate.csv', index_col=None,
                            header=0)
    risk_free_val = risk_free[['month', 'risk_free']]
    risk_free_val['risk_free'] /= 100
    risk_free_val = risk_free_val.set_index('month')
    risk_free_val += 1
    risk_free_val = risk_free_val.cumprod()
    risk_free_val = pd.Series(risk_free_val['risk_free'])
    # 输出结果
    all_val = pd.DataFrame({'long': long_val,
                            'short': short_val,
                            'market': market_val,
                            'risk-free': risk_free_val}, index=risk_free_val.index)
    all_val.to_csv(output_url + 'all_val.csv')

    return 0


# 结果可视化
def visualization(data):
    data = data[data.index >= '2002-01-01']
    plt.rcParams['font.family'] = 'Heiti TC'

    # # 策略年收益柱状图
    # x = data.index
    # y = data['hedge']
    # plt.figure(figsize=(20, 8))
    # plt.bar(x, y, width=0.8, color='black', alpha=0.8, label='Bar Chart')
    # plt.xticks(x[::2])
    # plt.xlabel('Year', fontsize=14)
    # plt.ylabel('Annual Returns', fontsize=14)
    # plt.show()

    # 策略投资1元的净值曲线
    data = np.log(data)
    x = data.index.map(lambda i: pd.to_datetime(i)).tolist()
    y1 = data['long']
    y2 = data['market']
    y3 = data['risk-free']
    y4 = data['short']
    plt.figure(figsize=(20, 8))
    plt.plot(x, y1, linestyle='-', color='black', label='Long Portfolio, Final Value is 57.66 Yuan')
    plt.plot(x, y2, linestyle='--', color='black', label='Market Portfolio，Final Value is 1.66 Yuan')
    plt.plot(x, y3, linestyle=':', color='black', label='Risk-free Rate, Final Value is 1.07 Yuan')
    plt.plot(x, y4, linestyle='-.', color='black', label='Short Portfolio, Final Value is 0.41 Yuan')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('log(Investment Net Value)', fontsize=14)
    plt.legend()
    plt.show()


def main():
    # # 导入数据
    # data = pd.read_csv('/Users/xuchuyu/PycharmProjects/Thesis/data/data.csv', header=0, index_col=0)
    # data['month'] = pd.to_datetime(data['month'])
    #
    # # 集中处理缺失值并统计数据分布
    # data = data_process(data)
    #
    # # 训练模型，预测
    # y_test_result = training_predicting(data)

    # # 策略评估绩效
    # # y_test_result = pd.read_csv('./result/y_test_result.csv')
    # strategy(y_test_result)

    # 结果可视化
    vs_data = pd.read_csv('/Users/xuchuyu/PycharmProjects/Thesis/result/all_val.csv', header=0, index_col=0)
    visualization(vs_data)


if __name__ == '__main__':
    main()
