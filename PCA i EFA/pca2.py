import pandas as pd
from scipy.cluster import hierarchy
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering

sports = pd.read_csv('C:/Users/marce/Desktop/biurowe 4/pp/genomika populacyjna/lab8/Decathlon Rio 2016.csv', decimal=',', sep=';', encoding='utf-8', encoding_errors='ignore')
sports = sports.iloc[:, 3:-1]

for name in sports.columns:
    if name[-1] == 's':
        sports[name] = (sports[name] - min(sports[name])) / (max(sports[name]) - min(sports[name]))
    else:
        sports[name] = (max(sports[name]) - sports[name]) / (max(sports[name]) - min(sports[name]))


clusters = hierarchy.linkage(sports, method="ward")
plt.figure(figsize=(8, 6))
dendrogram = hierarchy.dendrogram(clusters)
plt.axhline(1.6, color='red', linestyle='--')
plt.show()

model = AgglomerativeClustering(n_clusters=3, linkage="ward")
model.fit(sports)
labels = model.labels_
print(labels)


pca = PCA(n_components=2)  #4
principalComponents = pca.fit_transform(sports)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
principalDf['label'] = labels
print(principalDf)
#print(pca.explained_variance_ratio_)


loadings = pca.components_
n_features = pca.n_features_
feature_names = sports.columns
pc_list = [f'PC{i}' for i in list(range(1, n_features + 1))]
pc_loadings = dict(zip(pc_list, loadings))
loadings_df = pd.DataFrame.from_dict(pc_loadings)
loadings_df['feature_names'] = feature_names
loadings_df = loadings_df.set_index('feature_names')
print(loadings_df)



fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
colors = ['r', 'g', 'b']
targets = [0,1,2]
for target, color in zip(targets,colors):
    indicesToKeep = principalDf['label'] == target
    ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
               , principalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()
