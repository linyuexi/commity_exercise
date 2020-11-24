import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns

df = pd.read_csv(r"D:\GitHub\EnsembleLearning\HR_comma_sep.csv", index_col=None)
df = df.rename(columns={'satisfaction_level': 'satisfaction',
                        'last_evaluation': 'evaluation',
                        'number_project': 'projectCount',
                        'average_montly_hours': 'averageMonthlyHours',
                        'time_spend_company': 'yearsAtCompany',
                        'Work_accident': 'workAccident',
                        'promotion_last_5years': 'promotion',
                        'sales': 'department',
                        'left': 'turnover'})
front = df['turnover']
df.drop(labels=['turnover'], axis=1, inplace=True)
df.insert(0, 'turnover', front)

corr = df.corr()
sns.heatmap(data=corr, yticklabels=corr.columns.values, xticklabels=corr.index.values)
plt.show()

fig = plt.figure(figsize=(15, 4))
sns.kdeplot(x='satisfaction', data=df, hue='turnover', shade=True)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, \
    precision_recall_curve, roc_auc_score

df['department'] = df['department'].astype('category').cat.codes
df['salary'] = df['salary'].astype('category').cat.codes

target_name = 'turnover'
x = df.drop('turnover', axis=1)
y = df[target_name]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=123, stratify=y)

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

dtree = DecisionTreeClassifier(criterion='entropy', min_weight_fraction_leaf=0.01)
dtree.fit(x_train, y_train)
dt_roc_auc = roc_auc_score(y_test, dtree.predict(x_test))
print('决策树 AUC=%2.2f' % dt_roc_auc)
print(classification_report(y_test, dtree.predict(x_test)))

feature_names = df.columns[1:]
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                feature_names=feature_names, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())

importances = dtree.feature_importances_
feat_names = df.drop(['turnover'], axis=1).columns
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 6))
plt.title('Feature importances by Decision Tree')
plt.bar(range(len(indices)), importances[indices], color='lightblue', align='center')
plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical', fontsize=14)
plt.xlim([-1, len(indices)])
plt.show()
# --
rf = RandomForestClassifier(criterion='entropy', n_estimators=3, max_depth=None, min_samples_split=10)
rf.fit(x_train, y_train)
rf_roc_auc = roc_auc_score(y_test, rf.predict(x_test))
print('随机森林 AUC=%2.2f' % rf_roc_auc)
print(classification_report(y_test, rf.predict(x_test)))

Estimators = rf.estimators_
for index, model in enumerate(Estimators):
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data, feature_names=df.columns[1:],
                    class_names=['0', '1'], filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('RF{}.png'.format(index))
    plt.figure(figsize=(20, 20))
    plt.imshow(plt.imread('RF{}.png'.format(index)))
    plt.axis('off')
plt.show()

importances = rf.feature_importances_
feature_names = df.drop(['turnover'], axis=1, inplace=True).columns
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 6))
plt.title('Feature importances by RandomForest')
plt.bar(range(len(indices)), importances[indices], color='lightblue', align='center')
plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', lable='Cumulative')
plt.xticks(range(len(indices)), feat_names[indices], rotation=45, fontsize=14)
plt.show()

from sklearn.metrics import roc_curve

rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(x_test)[:, 1])
dt_fpr, dt_tpr, dt_thresholds = roc_curve(y_test, dtree.predict_proba(x_test)[:, 1])
plt.figure()
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area=%0.2f)' % rf_roc_auc)
plt.plot(dt_fpr, dt_tpr, label='Decision Tree (area=%0.2f)' % dt_roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc='best')
plt.show()
