import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import plotly.express as px
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from factor_analyzer.factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


sosna = pd.read_excel('./Sosna_Projekt1 (1).xls', header=1).fillna(0)
sosna.drop(['osobnik'], axis=1, inplace=True)

cols = [i for i in sosna.columns]
pops = set(sosna['populacja'])

sosna_means = sosna.groupby('populacja', sort=False).mean()
#sosna_means['populacja'] = sosna['populacja'].unique()
sosna_means_norm = stats.zscore(sosna_means)

sosna_norm = stats.zscore(sosna.iloc[:, :-1])

#### KLASTROWANIE ####
# clusters = hierarchy.linkage(sosna_means_norm, method="ward")
# plt.figure(figsize=(8, 6))
# dendrogram = hierarchy.dendrogram(clusters)
# plt.axhline(12.5, color='red', linestyle='--')
# plt.show()
#
# model = AgglomerativeClustering(n_clusters=3, linkage="ward")
# model.fit(sosna_means_norm)
# labels = model.labels_
# print(labels)
#
# for clust in range(3):
#     claster = [name for i, name in enumerate(pops) if labels[i] == clust]
#     print(f'Klaster {clust+1}: {set(claster)}')
#
#
# #
# Sc = StandardScaler()
# X = Sc.fit_transform(sosna_means_norm)
# pca = PCA(2)
# pca_data = pd.DataFrame(pca.fit_transform(X), columns=['PC1','PC2'])
# pca_data['cluster'] = pd.Categorical(labels)
# sns.scatterplot(x="PC1", y="PC2", hue="cluster", data=pca_data)
# plt.show()
#
#
#
# pca = PCA(3)
# principalComponents = pca.fit_transform(sosna_means_norm)
# principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
# principalDf['label'] = labels
# print(principalDf)
# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)
# print(pca.n_features_in_)
# print(pca.feature_names_in_)
#
# loadings = pca.components_
# n_features = pca.n_features_
# feature_names = sosna_means_norm.columns
# pc_list = [f'PC{i}' for i in list(range(1, n_features + 1))]
# pc_loadings = dict(zip(pc_list, loadings))
# loadings_df = pd.DataFrame.from_dict(pc_loadings)
# loadings_df['feature_names'] = feature_names
# loadings_df = loadings_df.set_index('feature_names')
# print(loadings_df)
#
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_zlabel('Principal Component 3', fontsize = 15)
# ax.set_title('3 component PCA', fontsize = 20)
# colors = ['r', 'g', 'b']
# targets = [0,1,2]
# for target, color in zip(targets,colors):
#     indicesToKeep = principalDf['label'] == target
#     ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
#                , principalDf.loc[indicesToKeep, 'principal component 2']
#                , principalDf.loc[indicesToKeep, 'principal component 3']
#                , c = color
#                , s = 50)
# ax.legend(targets)
# ax.grid()
# plt.show()

chi_square_value, p_value = calculate_bartlett_sphericity(sosna_means_norm)
print(f"Chisquare value: {chi_square_value}  pvalue: {p_value}")

fa = FactorAnalyzer(rotation='varimax', n_factors=6)
fa.fit(sosna_means_norm)
ev, v = fa.get_eigenvalues()
print(ev)

plt.scatter(range(1,sosna_means_norm.shape[1]+1),ev)
plt.plot(range(1,sosna_means_norm.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.axhline(y=1,c='k')
plt.show()


a = FactorAnalyzer(rotation="varimax", n_factors = 6)
fa.fit(sosna_means_norm)

new_dt = pd.DataFrame(fa.loadings_, columns=['Factor1', 'Factor2', 'Factor3', 'Factor4', 'Factor5', 'Factor6'], index=[sosna_means_norm.columns])
print(new_dt)

Z=np.abs(fa.loadings_)
fig, ax = plt.subplots()
c = ax.pcolor(Z)
fig.colorbar(c, ax=ax)
ax.set_yticks(np.arange(fa.loadings_.shape[0])+0.5, minor=False)
ax.set_xticks(np.arange(fa.loadings_.shape[1])+0.5, minor=False)
ax.set_yticklabels(sosna_means_norm.columns)
ax.set_xticklabels(['Factor1', 'Factor2', 'Factor3', 'Factor4', 'Factor5', 'Factor6'])
plt.show()





# ### REGRESJA WIELORAKA ####
# X = sosna[['ZM4', 'ZM11', 'ZM17', 'ZM20']]
# yy = sosna['ZM5']
#
# regr = linear_model.LinearRegression()
# regr.fit(X, yy)
#
# print(f'Skuteczność: {regr.score(X, yy)}')               #coefficient of determination of prediction (R^2)
# print(f'Współczynniki: {regr.coef_}')
# print(f'Współczynnik wolny (B0): {regr.intercept_}')







### WYKRESY ####
#sosna[sosna['populacja'] == 'AD'].iloc[:, 12:].plot(kind="box", subplots=True, layout=(3, 5), figsize=(13, 13))
#plt.show()


# fig = px.scatter_matrix(sosna_means, dimensions=cols[:-1], color='populacja')
# fig.update_traces(diagonal_visible=False)
# fig.show()

# sosna_norm = stats.zscore(sosna.iloc[:, :-1]).join(sosna['populacja'])
#
# print(sosna_norm.groupby('populacja').var())


# fig = plt.figure(figsize=(10, 7))
# plt.boxplot(sosna[sosna['populacja'] == 'AD']['ZM17'])
# plt.title('Boxplot dla cechy ZM15 w populacji AD')
# plt.show()

# fig, axe = plt.subplots(figsize=(4.3, 4.5))
# axe.scatter(sosna[sosna['populacja']=='AD']['ZM16'], sosna[sosna['populacja']=='AD']['ZM17'], alpha=0.8, color='green')
# axe.set_xlabel('ZM16', fontweight = 'light')
# axe.set_ylabel('ZM17', fontweight = 'light')
# plt.title('Zależnośc między cechami ZM16 i ZM17 w populacji AD')
# plt.show()
# plt.show()



# sosna_partial = sosna.iloc[:, :6]
# sosna_partial.join(sosna['populacja'])
#fig = px.scatter_matrix(sosna.iloc[:, :4].join(sosna['populacja']), dimensions=cols[:4], color='populacja')


# fig, axe = plt.subplots(figsize=(4.3, 4.5))
# axe.scatter(sosna[sosna['populacja']=='TR_10']['ZM1'], sosna[sosna['populacja']=='TR_10']['ZM2'], sosna[sosna['populacja']=='TR_10']['ZM3'])
# plt.show()

# fig = plt.figure(figsize = (12, 9))
# ax = plt.axes(projection = "3d")
#
# ax.grid(b = True, color ='grey',
#         linestyle ='-.', linewidth = 0.3,
#         alpha = 0.2)
#
# scatter = ax.scatter3D(sosna[sosna['populacja']=='TR_10']['ZM1'], sosna[sosna['populacja']=='TR_10']['ZM2'], sosna[sosna['populacja']=='TR_10']['ZM3'],
#              alpha=0.8, color='green')

# ax.set_xlabel('Płeć', fontweight = 'light')
# ax.set_ylabel('Wiek kierowcy', fontweight = 'light')
# ax.set_zlabel('Czas eksploatacji', fontweight = 'light')
#ax.set_xlim(left=1, right=0, emit=False)

# fig.text(0.1, 0.65, 'Legenda:\n0-Kobieta\n1-Mężczyzna', fontsize=16)

# plt.title('Zależnośc między płcią, wiekiem kierowcy a czasem eksploatacji samochodu')
# plt.show()
# plt.close(fig)
