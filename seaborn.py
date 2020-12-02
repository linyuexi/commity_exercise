import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

flights = sns.load_dataset('flights')

sns.lineplot(x='year', y='passengers', data=flights, hue='month', palette='tab20')
plt.show()

tips = sns.load_dataset("tips")
tips.head()

tips.plot.scatter(x='total_bill', y='tip', c='red')
plt.show()

sns.scatterplot(x='total_bill', y='tip', data=tips, hue='sex', style='smoker', size='size', sizes=(20, 200))
plt.show()

iris = sns.load_dataset('iris')
iris.head()
sns.scatterplot(x='petal_length', y='petal_width', data=iris, hue='species', style='species')
plt.show()

flights_table = flights.pivot(index='month', columns='year', values='passengers')

sns.heatmap(flights_table)
plt.show()

sns.rugplot(pd.Series(np.random.randn(100)))
plt.show()

sns.set()
sns.distplot(pd.Series(np.random.randn(100)), rug=True)
plt.show()

iris = sns.load_dataset('iris')
iris.head(2)
petal=iris[['petal_length',]]