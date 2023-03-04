import pandas as pd
from scipy.cluster import hierarchy
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering

crimes = pd.read_csv('./crime.csv', decimal=',', sep=';')
crimes_norm = stats.zscore(crimes.iloc[:, 1:])
print(crimes)

names = list(crimes.iloc[:, 0])
# #crimes.iloc[:, 0] = [i for i in range(len(crimes['Nazwy przyp.']))]

clusters = hierarchy.linkage(crimes_norm, method="ward")

plt.figure(figsize=(8, 6))
dendrogram = hierarchy.dendrogram(clusters)
plt.axhline(7.5, color='red', linestyle='--')
plt.show()

###to dodatkowo zeby zwizualizowac cala tabele w klastrach###
Sc = StandardScaler()
X = Sc.fit_transform(crimes_norm)
pca = PCA(2)
pca_data = pd.DataFrame(pca.fit_transform(X), columns=['PC1','PC2'])
###

model = AgglomerativeClustering(n_clusters=4, linkage="ward")
#model.fit(crimes_norm)
model.fit(X)
labels = model.labels_

pca_data['cluster'] = pd.Categorical(labels)
sns.scatterplot(x="PC1", y="PC2", hue="cluster", data=pca_data)
# fig,ax = plt.subplots()
# scatter = ax.scatter(pca_data['PC1'], pca_data['PC2'],c=pca_data['cluster'],cmap='Set3',alpha=0.7)
# legend1 = ax.legend(*scatter.legend_elements(),
#                     loc="upper left", title="")
# ax.add_artist(legend1)
plt.show()


labels, names = zip(*sorted(zip(labels, names)))

print(labels)
print(names)

first_claster = [name for i, name in enumerate(names) if labels[i] == 3]
srednie = crimes[crimes['Nazwy przyp.'].isin(first_claster)].iloc[:, 1:].mean()
print(first_claster)
print(srednie)

# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
# sns.scatterplot(ax=axes[0], data=crimes, x='RAPE', y='ROBBERY').set_title('Without cliustering')
# sns.scatterplot(ax=axes[1], data=crimes, x='RAPE', y='ROBBERY', hue=labels).set_title('With clustering')
# plt.show()

