import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

x1, y1 = make_gaussian_quantiles(cov=2., n_samples=200, n_features=2, n_classes=2, shuffle=True, random_state=1)
x2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5, n_samples=300, n_features=2, n_classes=2, shuffle=True,
                                 random_state=1)

X = np.vstack((x1, x2))
y = np.hstack((y1, 1 - y2))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

weakClassifier = DecisionTreeClassifier(max_depth=2)
clf = AdaBoostClassifier(base_estimator=weakClassifier, algorithm='SAMME', n_estimators=300, learning_rate=0.8)
clf.fit(X, y)

y_ = clf.predict(np.r_[x1,x2])
x1_min = X[:, 0].min() - 1
x1_max = X[:, 0].max() + 1
x2_min = X[:, 1].min() - 1
x2_max = X[:, 1].max() + 1
x1_, x2_ = np.meshgrid(np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02))
y_=y_.reshape(np.r_[x1,x2].shape)
plt.contourf(x1_, x2_, y_, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
