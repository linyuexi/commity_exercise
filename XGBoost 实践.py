import time
import numpy as np
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
import os

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123456)
params = {'booster': 'gbtree', 'nthread': 4, 'silent': 0, 'num_feature': 4,
          'seed': 1000, 'objective': 'multi:softmax', 'num_class': 3,
          'gamma': 0.1, 'max_depth': 6, 'lambda': 2, 'subsample': 0.7,
          'colsample_bytree': 0.7, 'min_child_weight': 3, 'eta': 0.1}

plst = params.items()
feature_name = iris.feature_names
dtrain = xgb.DMatrix(X_train, y_train, feature_names=feature_name)
dtest = xgb.DMatrix(X_test, feature_names=feature_name)
num_rounds = 50
model = xgb.train(plst, dtrain, num_rounds)
y_pred = model.predict(dtest)
accuracy = accuracy_score(y_test, y_pred)
print('accuracy:%.2f%%' % (accuracy * 100.0))
plot_importance(model)
plt.show()

plot_tree(model, num_trees=5)
plt.show()
# ---------
boston = load_boston()
X, y = boston.data, boston.target
feature_name = boston.feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
params = {'booster': 'gbtree',
          'objective': 'reg:gamma',
          'gamma': 0.1, 'max_depth': 5, 'lambda': 3, 'subsample': 0.7, 'colsample_bytree': 0.7, 'min_child_weight': 3,
          'silent': 1, 'eta': 0.1, 'seed': 1000, 'nthread': 4}
plst = params.items()
dtrain = xgb.DMatrix(X_train, y_train, feature_names=feature_name)
dtest = xgb.DMatrix(X_test, feature_names=feature_name)
num_rounds = 30
model = xgb.train(plst, dtrain, num_rounds)
y_pred = model.predict(dtest)
plot_importance(model, importance_type='weight')
plt.show()

plot_tree(model, num_trees=17)
plt.show()
# ------
iris = load_iris()
X, y = iris.data, iris.target
feature_name = iris.feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
model = xgb.XGBClassifier(max_depth=5, n_estimators=50, silent=True, objective='multi:softmax',
                          feature_names=feature_name)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('accuracy:%.2f%%' % (accuracy * 100.0))

plot_importance(model)
plt.show()

plot_tree(model, num_trees=5, feature_names=feature_name)
plt.show()
# ------------

boston = load_boston()
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=50, silent=True, objective='reg:gamma')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
plot_importance(model)
plt.show()

plot_tree(model, num_trees=17)
plt.show()
# ----
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

breast = load_breast_cancer()
X, y = breast.data, breast.target
feature_name = breast.feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test,  reference=lgb_train)

boost_round = 50
early_stop_rounds = 10
params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metric': {'12', 'auc'},
          'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.9, 'bagging_fraction': 0.8,
          'bagging_freq': 5, 'verbose': 1}
results = {}
gbm = lgb.train(params, lgb_train, num_boost_round=boost_round,
                valid_sets=(lgb_eval, lgb_train), valid_names=('validate', 'train'),
                early_stopping_rounds=early_stop_rounds, evals_result=results)
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

lgb.plot_metric(results)
plt.show()

lgb.plot_importance(gbm, importance_type='split')
plt.show()

lgb.plot_tree(gbm, tree_index=0)
plt.show()
