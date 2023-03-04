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

sports = pd.read_csv('C:/Users/marce/Desktop/biurowe 4/pp/genomika populacyjna/lab8/Decathlon Rio 2016.csv', decimal=',', sep=';', encoding='utf-8', encoding_errors='ignore')
sports = sports.iloc[:, 3:-1]

for name in sports.columns:
    if name[-1] == 's':
        sports[name] = (sports[name] - min(sports[name])) / (max(sports[name]) - min(sports[name]))
    else:
        sports[name] = (max(sports[name]) - sports[name]) / (max(sports[name]) - min(sports[name]))


chi_square_value, p_value = calculate_bartlett_sphericity(sports)
print(f"Chisquare value: {chi_square_value}  pvalue: {p_value}")

fa = FactorAnalyzer(rotation='varimax')
fa.fit(sports)
ev, v = fa.get_eigenvalues()

plt.scatter(range(1,sports.shape[1]+1),ev)
plt.plot(range(1,sports.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.axhline(y=1,c='k')
plt.show()

fa = FactorAnalyzer(rotation="varimax", n_factors = 4)
fa.fit(sports)

new_dt = pd.DataFrame(fa.loadings_, columns=['Factor1', 'Factor2', 'Factor3', 'Factor4'], index=[sports.columns])
print(new_dt)

Z=np.abs(fa.loadings_)
fig, ax = plt.subplots()
c = ax.pcolor(Z)
fig.colorbar(c, ax=ax)
ax.set_yticks(np.arange(fa.loadings_.shape[0])+0.5, minor=False)
ax.set_xticks(np.arange(fa.loadings_.shape[1])+0.5, minor=False)
ax.set_yticklabels(sports.columns)
ax.set_xticklabels(['Factor1', 'Factor2', 'Factor3', 'Factor4'])
plt.show()


another_dt = pd.DataFrame(fa.get_factor_variance(), columns=['Factor1', 'Factor2', 'Factor3', 'Factor4'], index=['SumSquare Loading', 'Prop Var', 'Cum Var'])
print(another_dt)