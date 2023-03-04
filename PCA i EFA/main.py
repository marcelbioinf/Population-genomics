import pandas as pd
from scipy.cluster import hierarchy
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering

personalities = pd.read_csv('C:/Users/marce/Desktop/biurowe 4/pp/genomika populacyjna/lab8/personalityTest.csv', decimal=',', sep=';')
personalities.iloc[:, 1:] = stats.zscore(personalities.iloc[:, 1:])
print(personalities)


clusters = hierarchy.linkage(personalities.iloc[:, 1:], method="ward")
plt.figure(figsize=(8, 6))
dendrogram = hierarchy.dendrogram(clusters)
plt.axhline(21, color='red', linestyle='--')
plt.show()

model = AgglomerativeClustering(n_clusters=4, linkage="ward")
model.fit(personalities.iloc[:, 1:])
labels = model.labels_
print(labels)
#personalities['cluster'] = labels
#print(personalities)
#means = personalities.iloc[:, 1:].groupby('cluster').mean()

# for i in range(means.shape[0]):
#      print(f"Min wartość klastru {i}: {min(means.iloc[i])}\nMaksymalna wartość klastru {i}: {max(means.iloc[i])}")
#      #print(means.apply(lambda row: row[row == min(means.iloc[i])].index, axis=1))
#      #print(means.apply(lambda row: row[row == max(means.iloc[i])].index, axis=1))
#      #print(means[means.iloc[i] == min(means.iloc[i])])
#      #print(means[means.iloc[i] == max(means.iloc[i])])
# print(f'Minimalna wartośc dla każdego klastra \n{means.idxmin(axis=1)}')
# print(f'Maksymalna wartośc dla każdego klastra \n{means.idxmax(axis=1)}')




pca = PCA(n_components=2)
principalComponents = pca.fit_transform(personalities.iloc[:, 1:])
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
principalDf['label'] = labels
print(principalDf)
# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)
# print(pca.n_features_in_)
# print(pca.feature_names_in_)


loadings = pca.components_
n_features = pca.n_features_
feature_names = personalities.iloc[:, 1:].columns
pc_list = [f'PC{i}' for i in list(range(1, n_features + 1))]
pc_loadings = dict(zip(pc_list, loadings))
loadings_df = pd.DataFrame.from_dict(pc_loadings)
loadings_df['feature_names'] = feature_names
loadings_df = loadings_df.set_index('feature_names')
print(loadings_df)






#################PLOT###################
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
colors = ['r', 'g', 'b', 'y']
targets = [0,1,2,3]
for target, color in zip(targets,colors):
    indicesToKeep = principalDf['label'] == target
    ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
               , principalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()












#
# ###PCA###
# Sc = StandardScaler()
# X = Sc.fit_transform(personalities.iloc[:, 1:])
# pca = PCA(2)
# pca_data = pd.DataFrame(pca.fit_transform(X), columns=['PC1','PC2'])
# ###
#
# model = AgglomerativeClustering(n_clusters=4, linkage="ward")
# model.fit(X)
# labels = model.labels_
#
# pca_data['cluster'] = pd.Categorical(labels)
# sns.scatterplot(x="PC1", y="PC2", hue="cluster", data=pca_data)
# plt.show()


# labels, names = zip(*sorted(zip(labels, names)))
#
# print(labels)
# print(names)
#
# first_claster = [name for i, name in enumerate(names) if labels[i] == 3]
# srednie = crimes[crimes['Nazwy przyp.'].isin(first_claster)].iloc[:, 1:].mean()
# print(first_claster)
# print(srednie)



