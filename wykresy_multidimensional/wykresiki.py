import pandas as pd
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


#ZAD2
# cars = pd.read_csv('./lab3/CarMarks.csv', decimal=',', sep=';')
# cols = []
# for col in cars.iloc[:, 1:].columns:
#     cols.append(col)
#
#
# fig = px.scatter_matrix(cars, dimensions=cols, color='Mark')
#
# fig.update_traces(diagonal_visible=False)
# fig.show()



#ZAD3
skladniki = pd.read_csv('./lab3/Składniki_wykres.csv', decimal=',', sep=';', engine='python')

fig = px.parallel_categories(skladniki)

fig.show()



#ZAD1
# polisy = pd.read_csv('./lab3/Polisy.csv', decimal=',', sep=';')
# polisy['Plec'].replace(['Kobieta', 'Mezczyzna'], [0.0, 1.0], inplace = True)
#
# fig = plt.figure(figsize = (12, 9))
# ax = plt.axes(projection = "3d")
#
# ax.grid(b = True, color ='grey',
#         linestyle ='-.', linewidth = 0.3,
#         alpha = 0.2)
#
# scatter = ax.scatter3D(polisy['Plec'], polisy['Wiek kierowcy'], polisy['Czas eksploatacji samochodu'],
#              alpha=0.8, color='green')
#
# ax.set_xlabel('Płeć', fontweight = 'light')
# ax.set_ylabel('Wiek kierowcy', fontweight = 'light')
# ax.set_zlabel('Czas eksploatacji', fontweight = 'light')
# #ax.set_xlim(left=1, right=0, emit=False)
#
# fig.text(0.1, 0.65, 'Legenda:\n0-Kobieta\n1-Mężczyzna', fontsize=16)
#
# plt.title('Zależnośc między płcią, wiekiem kierowcy a czasem eksploatacji samochodu')
# plt.show()
# plt.close(fig)
#
# # fig, axe = plt.subplots(figsize=(4.3, 4.5))
# # axe.scatter(polisy[polisy['Szkoda']=='Nie']['Wiek kierowcy'], polisy[polisy['Szkoda']=='Nie']['Czas eksploatacji samochodu'])
# # plt.show()
