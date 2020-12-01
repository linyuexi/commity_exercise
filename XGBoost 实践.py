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

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test)
num_rounds = 50
model = xgb.train(plst, dtrain, num_rounds)
y_pred = model.predict(dtest)
accuracy = accuracy_score(y_test, y_pred)
print('accuracy:%.2f%%' % (accuracy * 100.0))
plot_importance(model)
plt.show()

plot_tree(model,num_trees=5)
plt.show()
