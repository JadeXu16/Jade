import pandas as pd

data = pd.read_csv('/feature/initial_data.csv', header=0, index_col=None)
data_feature = pd.read_csv('/feature/initial_data.csv', header=0, index_col=None)
data['month'] = pd.to_datetime(data['month'])
data['code'] = data['code'].apply(lambda x: str(x).zfill(6))
data_feature['month'] = pd.to_datetime(data_feature['month'])
data_feature['code'] = data_feature['code'].apply(lambda x: str(x).zfill(6))

for i in range(25):
    data['conditional_month'] = data['month'].apply(lambda x: x - pd.DateOffset(months=i+1))
    data = pd.merge(data, data_feature, left_on=['conditional_month', 'code'], right_on=['month', 'code'], how='left')
    data = data.drop(['conditional_month', 'month_y'], axis=1)
    data = data.rename(columns={'month_x': 'month', 'ret_x': 'ret', 'ret_y': 'R_'+str(i)})

data.to_csv('./data.csv')
