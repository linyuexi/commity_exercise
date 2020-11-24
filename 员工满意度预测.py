import pandas as pd
import pandas_profiling as pdp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pprint import pprint
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, MinMaxScaler, MaxAbsScaler
from sklearn.decomposition import PCA

train_data = pd.read_csv(r"C:\Users\马朝阳\Desktop\常用数据\Employee_Satisfaction\训练集.csv")
test_data = pd.read_csv(r"C:\Users\马朝阳\Desktop\常用数据\Employee_Satisfaction\测试集.csv")
train_data.index = train_data.id
test_data.index = test_data.id
x_train = train_data.drop(columns=['satisfaction_level', 'id'], axis=1)
y_train = train_data['satisfaction_level']
x_test = test_data.drop('id', axis=1)


def encode(data, pca_comp_num=3):
    result = pd.DataFrame.copy(data, deep=True)
    division_le = OrdinalEncoder()
    package_le = OrdinalEncoder()
    salary_oe = OrdinalEncoder()

    result.division = division_le.fit_transform(result['division'].values.reshape(-1, 1))
    result.package = package_le.fit_transform(result['package'].values.reshape(-1, 1))
    result.salary = salary_oe.fit_transform(result['salary'].values.reshape(-1, 1))

    for col in ['last_evaluation', 'average_monthly_hours']:
        maxAbsEnc = MaxAbsScaler()
        result[col] = maxAbsEnc.fit_transform(result[col].values.reshape(-1, 1))
    for col in ['number_project', 'time_spend_company', 'package', 'division']:
        pca = PCA(n_components=pca_comp_num)
        new_col = pca.fit_transform(pd.get_dummies(data=result[col]).values.reshape(-1, 1))
        for i in range(pca_comp_num):
            result[col + '_' + str(i)] = new_col[:, i]
        result.drop(columns=[col], axis=1, inplace=True)
    for col in ['Work_accident', 'promotion_last_5years', 'salary']:
        one_hot_encode = pd.get_dummies(data=result[col])
        one_hot_encode.columns = [col + '_' + str(i) for i in range(len(one_hot_encode.columns))]
        result = result.join(one_hot_encode)
        result.drop(col, axis=1, inplace=True)
    return result


x_test_cleaned = encode(x_test, 4)
x_train_cleaned = encode(x_train, 4)
