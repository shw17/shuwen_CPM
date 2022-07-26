import pickle
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime

warnings.filterwarnings('ignore')


# RENDER_NBUMBER
def percentile_encoding(df, column_name):
    bins_height = [df[column_name].min(), df[column_name].quantile(.25)+2, df[column_name].quantile(.50)+3,
                   df[column_name].quantile(.75)+4, df[column_name].max()]
    category = ['lower', 'low', 'medium', 'high']
    # convert numerical to category
    df[column_name] = pd.cut(df[column_name], bins_height, labels=category, include_lowest=True)
    # label encoding
    df[f'{column_name}_ENCODE'] = df[column_name].astype('category').cat.codes
    return df


# AD_TYPE, DAY
def one_hot_encoding(df, column_name):
    df = pd.get_dummies(df, columns=column_name, drop_first=True)
    return df


# COUNTRY_GROUP, BROWSER_GROUP, DEVICE_TYPE
def frequency_encoding(df, column_name):
    fe = df.groupby(column_name).size()
    co_ = fe / df.shape[0]
    df[f'{column_name}_FE'] = df[column_name].map(co_).round(2)
    return df


# UNIT
def label_encoding(df, column_name):
    df[f'{column_name}_ENCODE'] = df[column_name].astype('category').cat.codes()
    return df


# drop and sqrt CPM
def drop_function(df, column_name):
    ppp = df.pop('CPM')
    df.insert(df.shape[1], 'CPM', ppp)
    df['CPM'] = df['CPM'] ** 0.5
    df = df.drop(column_name, axis=1)
    return df




















# df['COUNTRY_FE'] = df['COUNTRY_GROUP'].apply(lambda x: country_fe[x])
# df = pd.read_csv('/Users/shuwen/Desktop/PLAYWIRE/DATASETS/arkid.csv')
# print(df.UNIT.value_counts())

# df['BID_DATE'] = pd.to_datetime(df['BID_DATE'])
# df = df[df['BID_DATE'] >= '2022-07-01']
# df = df.sort_values(by='BID_DATE').reset_index(drop=True)
#
# bins_height = [df.RENDER_NUMBER.min(), df.RENDER_NUMBER.quantile(.25), df.RENDER_NUMBER.quantile(.50),
#         df.RENDER_NUMBER.quantile(.75), df.RENDER_NUMBER.max()]
# category = ['lowest', 'low', 'medium', 'high']
# df['RENDER'] = pd.cut(df['RENDER_NUMBER'], bins_height, labels=category,include_lowest=True)
# df['RENDER_ENCODE'] = df['RENDER'].astype('category').cat.codes
#
# bins_height = [df.CONNECTION_DOWNLINK.min(), df.CONNECTION_DOWNLINK.quantile(.25)+1, df.CONNECTION_DOWNLINK.quantile(.50)+2,
#         df.CONNECTION_DOWNLINK.quantile(.75)+3, df.CONNECTION_DOWNLINK.max()+4]
# category = ['lowest', 'low', 'medium', 'high']
# df['CONNECTION'] = pd.cut(df['CONNECTION_DOWNLINK'], bins_height, labels=category,include_lowest=True)
# df['CONNECTION_ENCODE'] = df['CONNECTION'].astype('category').cat.codes
#
# country_fe = df.groupby('COUNTRY_GROUP').size()
# co_ = country_fe / df.shape[0]
# df['COUNTRY_FE'] = df['COUNTRY_GROUP'].map(co_).round(2)
# #
# browser_fe = df.groupby('BROWSER_GROUP').size()
# co_ = browser_fe / df.shape[0]
# df['BROWSER_FE'] = df['BROWSER_GROUP'].map(co_).round(2)
# # #
# device_fe = df.groupby('DEVICE_TYPE').size()
# co_ = device_fe / df.shape[0]
# df['DEVICE_FE'] = df['DEVICE_TYPE'].map(co_).round(2)
# # #
# os_fe = df.groupby('OS_NAME').size()
# co_ = os_fe / df.shape[0]
# df['OS_FE'] = df['OS_NAME'].map(co_).round(2)
# #
#
# df = pd.get_dummies(df, columns=['REFERRER'],
#                     drop_first=True)
#
# df['UNIT_ENCODE'] = df['UNIT'].astype('category').cat.codes
# df['UNIT_SUB_ENCODE'] = df['UNIT_SUBTYPE'].astype('category').cat.codes
# #
# df = df
# df = df.drop(['COUNTRY_GROUP', 'UNIT', 'DEVICE_TYPE', 'UNIT_SUBTYPE', 'WINDOW_WIDTH', 'WINDOW_HEIGHT',
#                                 'BROWSER_GROUP', 'OS_NAME', 'DAY', 'RENDER', 'RENDER_NUMBER', 'CONNECTION','CONNECTION_DOWNLINK'], axis=1)
# df['NEW_TIME'] = df.HOUR + (df.DAY_OF_MONTH - 1) * 24
# df = df.drop(['DAY_OF_MONTH', 'WEEK_OF_MONTH', 'HOUR'], axis=1)
# ppp = df.pop('CPM')
# df.insert(df.shape[1], 'CPM', ppp)
# pp = df.pop('BID_DATE')
# df.insert(df.shape[1], 'BID_DATE', pp)
# df['CPM'] = df['CPM'] ** 0.5
# #
# # pred = loaded_model.predict(df.iloc[:, :-2])
#
# deepth = [i for i in range(1,10)]
# r2 = []
#
# r2gb = []
# r2_testgb = []
# for i in deepth:
#     forest = GradientBoostingRegressor(max_depth=i)
#     _ = forest.fit(X_train, y_train)
#     r2_train = forest.score(X_train, y_train)
#     r2_test= forest.score(df.iloc[:, :-2], df.iloc[:, -2])
#     pred = forest.predict(df.iloc[:, :-2])
#     rmse = mean_squared_error(df.iloc[:, -2], pred)
#     r2.append(r2_train)
#     r2_testgb.append(r2_test)
#     r2gb.append(rmse)
#
#
# pyplot.figure(1)
# pyplot.plot(deepth, r2, '-o', label='Train')
# pyplot.plot(deepth, r2_testgb, '-o', label='Test')
# pyplot.plot(deepth, r2gb, '-*', label='mse')
# pyplot.title('gb')
# pyplot.legend()
# pyplot.show()

# # save and load the model
# pickle.dump(forest, open("arkidnew.pickle.dat", "wb"))
# loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
# pred = forest.predict(X_test)
# rrr = forest.score(X_test, y_test)
# print(rrr)
# print(mean_squared_error(y_test, pred, squared=False))
