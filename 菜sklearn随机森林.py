import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

dataset = load_boston()
print(dataset.data.shape)
x_full, y_full = dataset.data, dataset.target
n_samples = x_full.shape[0]
n_features = x_full.shape[1]
rng = np.random.RandomState(0)
missing_rate = 0.5
n_missing_samples = int(np.floor(n_samples * n_features * missing_rate))
missing_features = rng.randint(0, n_features, n_missing_samples)
missing_samples = rng.randint(0, n_samples, n_missing_samples)

X_missing = x_full.copy()
y_missing = y_full.copy()
X_missing[missing_samples, missing_features] = np.nan
X_missing = pd.DataFrame(X_missing)

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
x_missing_mean = imp_mean.fit_transform(X_missing)

imp_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
x_missing_0 = imp_0.fit_transform(X_missing)

X_missing_reg = X_missing.copy()
sortindex = np.argsort(X_missing_reg.isnull().sum(axis=0)).values
for i in sortindex:
    df = X_missing_reg
    fillc = df.iloc[:, i]
    df = pd.concat([df.iloc[:, df.columns != i], pd.DataFrame(y_full)], axis=1)
    df_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(df)
    Ytrain = fillc[fillc.notnull()]
    Ytest = fillc[fillc.isnull()]
    Xtrain = df_0[Ytrain.index, :]
    Xtest = df_0[Ytest.index, :]
    rfc = RandomForestRegressor(n_estimators=100)
    rfc = rfc.fit(Xtrain, Ytrain)
    Ypredict = rfc.predict(Xtest)
    X_missing_reg.loc[X_missing_reg.iloc[:, i].isnull(), i] = Ypredict
X = [x_full, x_missing_mean, x_missing_0, X_missing_reg]
mse = []
for x in X:
    estimator = RandomForestRegressor(random_state=0, n_estimators=100)
    scores = cross_val_score(estimator, x, y_full, scoring='neg_mean_squared_error', cv=5).mean()
    mse.append(scores * -1)

x_labels = ['Full Data', 'Mean Imputation', 'Zero Imputation', 'Regressor Imputation']
colors = ['r', 'g', 'b', 'orange']
plt.figure()
ax = plt.subplot(111)
for i in np.arange(len(mse)):
    ax.barh(i, mse[i], color=colors[i], alpha=0.6, align='center')
ax.set_title('Imputation Techniques with Boston Data')
ax.set_xlim(left=np.min(mse) * 0.9, right=np.max(mse) * 1.1)
ax.set_yticks(np.arange(len(mse)))
ax.set_label('MSE')
ax.set_yticklabels(x_labels)
plt.show()

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

sort_index = np.argsort(data1.isnull().sum(axis=0)).values
for i in sort_index:
    df = data1
    fillc = df.loc[:, i]
    df = df.loc[:, df.columns != i]
    df_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(df)
    y_train1 = fillc[fillc.notnull()]
    y_test1 = fillc[fillc.isnull()]
    X_train1 = df_0[y_train1.index, :]
    X_test1 = df_0[y_test1.index, :]
    rfc = RandomForestRegressor()
    rfc.fit(X_train1, y_train1)
    y_predict1 = rfc.predict(X_test1)
    data1.loc[data1.loc[:, i].isnull(), i] = y_predict1

