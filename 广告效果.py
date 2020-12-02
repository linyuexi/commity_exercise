#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

mpl.rcParams['font.sans-serif'] = ['SimiHei']
mpl.rcParams['axes.unicode_minus'] = False
raw_data = pd.read_csv(r"C:\Users\马朝阳\Desktop\常用数据\guanggao2482\ad_performance.csv")
raw_data.head()

# In[2]:


raw_data.describe(include='all')

# In[3]:


raw_data2 = raw_data.drop(['平均停留时间'], axis=1)
cols = ["素材类型", "广告类型", "合作方式", "广告尺寸", "广告卖点"]
for x in cols:
    data = raw_data2[x].unique()
    print('变量[{0}]的取值有:\n{1}'.format(x, data))
    print('-.' * 20)

# In[7]:


model_ohe = OneHotEncoder(sparse=True)
ohe_matrix = model_ohe.fit_transform(raw_data2[cols])

pd.read_csv()

sacle_matrix = raw_data2.iloc[:, 1:7]
model_scaler = MinMaxScaler()
data_scaled = model_scaler.fit_transform(sacle_matrix)

X = np.hstack((data_scaled, ohe_matrix))

score_list = []
silhouette_int = -1
for n_clusters in range(2, 8):
    model_kmeans = KMeans(n_clusters=n_clusters)
    labels_tmp = model_kmeans.fit_predict(X)
    silhouette_tmp = silhouette_score(X, labels_tmp)
    if silhouette_tmp > silhouette_int:
        best_k = n_clusters
        silhouette_int = silhouette_tmp
        best_kmeans = model_kmeans
        cluster_labels_k = labels_tmp
    score_list.append([n_clusters, silhouette_tmp])
print('{:*^60}'.format('k值对应的轮廓系数：'))
print(np.array(score_list))
print('最优的K值是:{}\n', best_k, silhouette_int)

cluster_features = []
for line in range(best_k):
    label_data = merge_data[merge_data['clusters'] == line]
    part1_data = label_data.iloc[:, 1:7]
    part1_desc = part1_data.describe().round(3)
    merge_data1 = part1_desc.iloc[2, :]
    part2_data = label_data.iloc[:, 7:-1]
    part2_desc = part2_data.describe(include='all')
    merge_data2 = part2_desc.iloc[2, :]
    merge_line = pd.concat((merge_data1, merge_data2), axis=0)
    cluster_features.append(merge_line)
cluster_pd = pd.DataFrame(cluster_features).T
all_cluster_set = pd.concat((clustering_count, clustering_ratio, cluster_pd), axis=0)

num_sets = cluster_pd.iloc[:, 6].T.astype(np.float64)
num_sets_max_min = model_scaler.fit_transform(num_sets)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)
labels = np.array(merge_data1.index)
cor_list = ['g', 'r', 'y', 'b']
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

for i in range(len(num_sets)):
    data_tmp = num_sets_max_min[i, :]
    data = np.concatenate((data_tmp, [data_tmp[0]]))
    ax.plot(angles, data, 'o-', c=cor_list[i], label='第%d类渠道' % (i))
    ax.fill(angles, data, alpha=2.5)
ax.set_thetagrids(angles * 180 / np.pi, labels, fontproperties='SimHei')
ax.set_title('各聚类类别显著特征对比', fontproperties='SimHei')
ax.set_rlim(-0.2, 1.2)
ptl.legend()
