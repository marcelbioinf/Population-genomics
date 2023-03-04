import pandas as pd
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

zoo = pd.read_csv('./zoo.csv', sep=';')

class_types = list(zoo.iloc[:, -1].unique())
class_types_dict = dict(zip(class_types, [i+1 for i in range(len(class_types))]))

zoo['class_type'] = zoo['class_type'].map(class_types_dict)
names = list(zoo.iloc[:, 0])
print(zoo)

clusters = hierarchy.linkage(zoo.iloc[:, 1:-1], method="ward")

plt.figure(figsize=(8, 6))
dendrogram = hierarchy.dendrogram(clusters)
plt.axhline(12, color='red', linestyle='--')
plt.show()

model = AgglomerativeClustering(n_clusters=4, linkage="ward")
model.fit(zoo.iloc[:, 1:-1])
labels = model.labels_
print(labels)

for clust in range(4):
    claster = [name for i, name in enumerate(names) if labels[i] == clust]
    print(f'Klaster {clust+1}: {claster}')