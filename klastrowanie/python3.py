import pandas as pd
from scipy.cluster import hierarchy
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
#pd.set_option('display.max_columns',10)

skladniki = pd.read_csv('./Składniki.csv', sep=';', decimal=',')
ingredients = list(skladniki.iloc[:, 0])

groups = list(skladniki['Group'].unique())
dic = dict(zip(groups, [i+1 for i in range(len(groups))]))
skladniki['Group'] = skladniki['Group'].map(dic)
skladniki = skladniki.fillna(value=0.0)

skladniki.iloc[:, 3:] = stats.zscore(skladniki.iloc[:, 3:])
print(skladniki)

clusters = hierarchy.linkage(skladniki.iloc[:, 3:], method="ward")
plt.figure(figsize=(8, 6))
dendrogram = hierarchy.dendrogram(clusters)
plt.axhline(11, color='red', linestyle='--')
plt.show()


model = AgglomerativeClustering(n_clusters=5, linkage="ward")
model.fit(skladniki.iloc[:, 3:])
labels = model.labels_
print(labels)

for clust in range(5):
    claster = [name for i, name in enumerate(ingredients) if labels[i] == clust]
    print(f'Klaster {clust+1}: {claster}')