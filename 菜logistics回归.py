from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv(r"D:\文件\接收集\资料\菜菜的sklearn课堂课件\05逻辑回归与评分卡\rankingcard.csv", index_col=0)
data.drop_duplicates(inplace=True)
data.index = range(data.shape[0])


def fill_missing_rf(X, y, to_fill):
    df = X.copy()
    fill = df.loc[:, to_fill]
    df = pd.concat([df.loc[:, df.columns != to_fill], pd.DataFrame(y)], axis=1)
    # 找出我们的训练集和测试集
    Ytrain = fill[fill.notnull()]
    Ytest = fill[fill.isnull()]
    Xtrain = df.iloc[Ytrain.index, :]
    Xtest = df.iloc[Ytest.index, :]
    # 用随机森林回归来填补缺失值
    from sklearn.ensemble import RandomForestRegressor as rfr

    rfr = rfr(n_estimators=100)
    rfr = rfr.fit(Xtrain, Ytrain)
    Ypredict = rfr.predict(Xtest)
    return Ypredict


X_train, X_test, y_train, y_test = train_test_split(
    X1, y1, test_size=0.25)
import imblearn
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)

X_train, y_train = sm.fit_sample(X_train, y_train)
X_test, y_test = sm.fit_sample(X_test, y_test)

n_sample_ = X.shape[0]
pd.Series(y).value_counts()
n_1_sample = pd.Series(y).value_counts()[1]
n_0_sample = pd.Series(y).value_counts()[0]
print('样本个数：{}; 1占{:.2%}; 0占{: .2 %}'.format(n_sample_, n_1_sample / n_sample_, n_0_sample / n_sample_))

from sklearn.feature_selection import SelectFromModel

X_embedded = SelectFromModel(clf, norm_order=1).fit(X_train, y_train)
X_embedded.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = data.data
y = data.target

l2 = []
l2test = []
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=0)
for i in np.arange(1, 201, 10):
    lrl2 = LogisticRegression(penalty='l2', solver='liblinear', C=0.9, max_iter=i)
    lrl2.fit(Xtrian, Ytrain)
    l2.append(accuracy_score(Ytrain, lrl2.predict(Xtrain)))
    l2test.append(accuracy_score(Ytest, lrl2.predict(Xtest)))
graph = [l2, l2test]
color = ['gray', 'green']
label = ['l2', 'l2test']
plt.figure(figsize=(20, 5))
for i in range(len(graph)):
    plt.plot(np.arange(1, 201, 10), graph[i], color[i], label=label[i])
    plt.legend(loc=4)
    plt.xticks(np.arange(1, 201, 10))
    plt.show()
lr = LogisticRegression(penalty='l2', solver='liblinear', C=0.9, max_iter=300).fit(Xtrain, Ytrain)
print(lr.n_iter_)
