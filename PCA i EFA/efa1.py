import pandas as pd
from scipy.cluster import hierarchy
from scipy import stats
from factor_analyzer.factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering

personalities = pd.read_csv('C:/Users/marce/Desktop/biurowe 4/pp/genomika populacyjna/lab8/personalityTest.csv', decimal=',', sep=';')
#personalities.iloc[:, 1:] = stats.zscore(personalities.iloc[:, 1:])
personalities.drop(['ID przypadku'], axis=1, inplace=True)
print(personalities)

chi_square_value, p_value = calculate_bartlett_sphericity(personalities)
print(f"Chisquare value: {chi_square_value}  pvalue: {p_value}")

fa = FactorAnalyzer(rotation='varimax')
fa.fit(personalities)
ev, v = fa.get_eigenvalues()
print(ev)

plt.scatter(range(1,personalities.shape[1]+1),ev)
plt.plot(range(1,personalities.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.axhline(y=1,c='k')
plt.show()

fa = FactorAnalyzer(rotation="varimax", n_factors = 6)
fa.fit(personalities)

new_dt = pd.DataFrame(fa.loadings_, columns=['Factor1', 'Factor2', 'Factor3', 'Factor4', 'Factor5', 'Factor6'], index=[personalities.columns])
print(new_dt)

Z=np.abs(fa.loadings_)
fig, ax = plt.subplots()
c = ax.pcolor(Z)
fig.colorbar(c, ax=ax)
ax.set_yticks(np.arange(fa.loadings_.shape[0])+0.5, minor=False)
ax.set_xticks(np.arange(fa.loadings_.shape[1])+0.5, minor=False)
ax.set_yticklabels(personalities.columns)
ax.set_xticklabels(['Factor1', 'Factor2', 'Factor3', 'Factor4', 'Factor5', 'Factor6'])
plt.show()


another_dt = pd.DataFrame(fa.get_factor_variance(), columns=['Factor1', 'Factor2', 'Factor3', 'Factor4', 'Factor5', 'Factor6'], index=['SumSquare Loading', 'Prop Var', 'Cum Var'])
print(another_dt)