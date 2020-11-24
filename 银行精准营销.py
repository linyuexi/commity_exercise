import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn import preprocessing

df_bankmarketing = pd.read_csv(r"C:\Users\马朝阳\Desktop\常用数据\bank-additional\bank-additional-full.csv", delimiter=';')
print(df_bankmarketing.head())
for col in df_bankmarketing.columns:
    if type(df_bankmarketing[col][0]) is str:
        print('unknown value count in ' + col + ':' + str(
            df_bankmarketing[df_bankmarketing[col] == 'unknown']['y'].count()))
# job与marital中的unknown的个数较少，可以直接做删除处理
# 其余特征的unknown作为该列的一个取值保留
for i in range(0, len(df_bankmarketing)):
    if df_bankmarketing.loc[i, 'job'] == 'unknown':
        df_bankmarketing.loc[i, 'job'] = np.nan
    if df_bankmarketing.loc[i, 'marital'] == 'unknown':
        df_bankmarketing.loc[i, 'marital'] = np.nan
df_bankmarketing = df_bankmarketing.dropna()
df_bankmarketing.index = range(0, len(df_bankmarketing))
df_bankmarketing = pd.get_dummies(df_bankmarketing)

import re

re.findall(r'fkit', 'Fkit is a good domain,FKIT is good')
a = re.compile(r"""020
\-
\d{8},re.X""")
b = re.compile(r'020\-\d{8}')

re.search(r'Windows (95|98|NT|2000)[\w]+\1','Windows 98 published inn 98')