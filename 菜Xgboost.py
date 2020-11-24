from xgboost import XGBRegressor as XGBR

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import datetime

data = load_boston()
# 波士顿数据集非常简单，但它所涉及到的问题却很多

X = data.data
y = data.target

Xtrain, Xtest, Ytrain, Ytest = TTS(X, y, test_size=0.3, random_state=420)

var = []
ge = []
for i in axisx:
    reg = XGBR(n_estimators=i, random_state=0)
    cvresult = CVS(reg, Xtrain, Ytrain, cv=5)
    rs.append(cvresult.mean())
    var.append(cvresult.var())
    ge.append((1 - cvresult.mean()) ** 2 + cvresult.var())
print(axisx[rs.index(max(rs))], max(rs), var[rs.index(max(rs))])
print(axisx[var.index(min(var))], rs[var.index(min(var))], min(var))
print(axisx[ge.index(min(ge))], rs[ge.index(min(ge))], var[ge.index(min(ge))], min(ge))

plt.figure(figsize=(20, 5))
plt.plot(axisx, rs, c='red', label='XGB')
plt.legend()
plt.show()

rs = np.array(rs)
var = np.array(var) * 0.01
plt.figure(figsize=(20, 5))
plt.plot(axisx, rs, c='black', label='XGB')
plt.plot(axisx, rs + var, c='red', linestyle='-.')
plt.plot(axisx, rs - var, c='red', linestyle='-.')
plt.legend()
plt.show()

plt.figure(figsize=(20, 5))
plt.plot(axisx, ge, c='gray', linestyle='-.')
plt.show()

reg = XGBR(n_estimators=100).fit(Xtrain, Ytrain)


def regassess(reg, Xtrain, Ytrain, cv, scoring=['r2'], show=True):
    score = []
    for i in range(len(scoring)):
        if show:
            print('{}:{:.2f}'.format(scoring[i], CVS(reg, Xtrain, Ytrain, cv=cv, scoring=scoring[i]).mean()))
            score.append(CVS(reg, Xtrain, Ytrain, cv=cv, scoring=scoring[i]).mean())
    return score


from time import time
import datetime

for i in [0, 0.2, 0.5, 1]:
    reg = XGBR(n_estimators=100, learning_rate=i)
    print('learning_rate={}'.format(i))
    regassess(reg, Xtrain, Ytrain, 5, scoring=['r2', 'neg_mean_squared_error'])

axisx = np.arange(0.05, 1, 0.05)
rs = []
te = []
for i in axisx:
    reg = XGBR(n_estimators=180, learning_rate=i)
    score = regassess(reg, Xtrain, Ytrain, cv=5, scoring=['r2', 'neg_mean_squared_error'], show=False)
    test = reg.fit(Xtrain, Ytrain).score(Xtest, Ytest)
    print(score)
#     rs.append(score)
#     te.append(test)
# print(axisx[rs.index(max(rs))], max(rs))
# plt.figure(figsize=(20, 10))
# plt.plot(axisx, te, c='gray', label='XGB')
# plt.plot(axisx, rs, c='green', label='XGB')
# plt.legend()
# plt.show()

for booster in ['gbtree', 'gblinear', 'dart']:
    reg = XGBR(n_estimators=180,
               learning_rate=0.1,
               random_state=0,
               booster=booster,
               objective='reg:squarederror').fit(Xtrain, Ytrain)
    print(booster)
    print(reg.score(Xtest, Ytest))

reg = XGBR(n_estimators=180, objective='reg:squarederror').fit(Xtrain, Ytrain)
reg.score(Xtest, Ytest)
MSE(Ytest, reg.predict(Xtest))

import xgboost as xgb

dtrain = xgb.DMatrix(Xtrain, Ytrain)
dtest = xgb.DMatrix(Xtest, Ytest)

param = {'silent': False, 'objective': 'reg:squarederror', 'eta': 0.1}
num_round = 180
bst = xgb.train(param, dtrain, num_round)
from sklearn.metrics import r2_score

axisx = np.arange(0, 5, 0.05)
rs = []
var = []
ge = []
for i in axisx:
    reg = XGBR(objective='reg:squarederror', n_estimators=180, gamma=i)
    result = CVS(reg, Xtrain, Ytrain, cv=5)
    rs.append(result.mean())
    var.append(result.var())
    ge.append((1 - result.mean()) ** 2 + result.var())
print(axisx[rs.index(max(rs))], max(rs), var[rs.index(max(rs))])
print(axisx[var.index(min(var))], rs[var.index(min(var))], min(var))
print(axisx[ge.index(min(ge))], rs[ge.index(min(ge))], var[ge.index(min(ge))], min(ge))
rs = np.array(rs)
var = np.array(var) * 0.1
plt.figure(figsize=(20, 10))
plt.plot(axisx, rs, c='black', label='XGB')
plt.plot(axisx, rs + var, c='red', linestyle='-.')
plt.plot(axisx, rs - var, c='red', linestyle='-.')
plt.legend()
plt.show()

import xgboost as xgb

dfull = xgb.DMatrix(X, y)
param1 = {'silent': True, 'obj': 'reg:squarederror', 'gamma': 0}
param2 = {'silent': True, 'obj': 'reg:squarederror', 'gamma': 20}

num_round = 180
n_fold = 5

cvresult1 = xgb.cv(param1, dfull, num_round, n_fold)
cvresult2 = xgb.cv(param2, dfull, num_round, n_fold)
plt.figure(figsize=(20, 10))
plt.grid()
plt.plot(range(1, 181), cvresult1.iloc[:, 0], c='red', label='train,gamma=0')
plt.plot(range(1, 181), cvresult1.iloc[:, 2], c='orange', label='test,gamma=0')
plt.plot(range(1, 181), cvresult2.iloc[:, 0], c='green', label='train,gamma=20')
plt.plot(range(1, 181), cvresult2.iloc[:, 2], c='blue', label='test,gamma=20')
plt.legend()
plt.show()

from sklearn.datasets import load_breast_cancer

data2 = load_breast_cancer()
x2 = data2.data
y2 = data2.target
dfull2 = xgb.DMatrix(x2, y2)
param1 = {'silent': True, 'obj': 'binary:logistic', 'gamma': 0, 'nfold': 5}
param2 = {'silent': True, 'obj': 'binary:logistic', 'gamma': 2, 'nfold': 5}
num_round = 100
cvresult1 = xgb.cv(param1, dfull2, num_round, metrics=('error'))
cvresult2 = xgb.cv(param2, dfull2, num_round, metrics=('error'))

plt.figure(figsize=(20, 10))
plt.grid()
plt.plot(range(1, 101), cvresult1.iloc[:, 0], c='red', label='train,gamma=0')
plt.plot(range(1, 101), cvresult1.iloc[:, 2], c='orange', label='test,gamma=0')
plt.plot(range(1, 101), cvresult2.iloc[:, 0], c='green', label='train,gamma=2')
plt.plot(range(1, 101), cvresult1.iloc[:, 2], c='blue', label='test,gamma=2')
plt.legend()
plt.show()

from sklearn.ensemble import RandomForestClassifier

fillc = data.loc[:, 'U_MIZE']
df = data.loc[:, [k for k in columns1 if k != 'U_MIZE']]
df['IS_DUAL'].fillna(df['IS_DUAL'].mode()[0], inplace=True)
df['CUST_SEX'].fillna(df['CUST_SEX'].mode()[0], inplace=True)
y_train = fillc[fillc.notnull()]
y_test = fillc[fillc.isnull()]
X_train = df.loc[ytrain.index, :].values
X_test = df.loc[ytest.index, :].values

rfc = RandomForestClassifier(n_estimators=20, n_jobs=3)
rfc.fit(Xtrain, y_train.values).astype('int')
y_predict = rfc.predict(Xtest)
data.loc[data.loc[:, 'U_MIZE'].isnull(), 'U_MIZE'] = y_predict;

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)

from sklearn.linear_model import LogisticRegressionCV

# multinomial和ovr的确定
for solver in ['lbfgs', 'newton-cg', 'sag', 'saga']:
    for multi_class in ('multinomial', 'ovr'):
        clf = LogisticRegressionCV(solver=solver, max_iter=1000, multi_class=multi_class)
        clf.fit(X, y)
        score = clf.score(X_test, y_test)
        print(solver, multi_class, 'training score:%.3f' % (score))


