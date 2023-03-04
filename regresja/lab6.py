import pandas as pd
from sklearn import linear_model
from matplotlib import pyplot as plt
from matplotlib import style

# df = pd.read_csv('./Chemical_Companies.csv', decimal=',', sep=';')
#
# X = df[['D/E', 'PAYOUTR1', 'NMP1', 'SALESGR5']]
# y = df['P/E']
#
# regr = linear_model.LinearRegression()
# regr.fit(X, y)
#
# print(f'Skuteczność: {regr.score(X, y)}')               #coefficient of determination of prediction (R^2)
# print(f'Współczynniki: {regr.coef_}')
# print(f'Współczynnik wolny (B0): {regr.intercept_}')
#
#
# predicted = regr.predict([[0.5, 0.32, 4.4, 18.3]])
# print(f'Przewidywwana wartość: {predicted}')
#
# style.use("ggplot")
# plt.figure(figsize=(12,22))
# for x, param in enumerate(X):
#     plt.subplot(2, 2, x+1)
#     plt.scatter(df[param], y)
#     plt.xlabel(param)
#     plt.ylabel(y.name)
# plt.show()


df = pd.read_csv('./CarMarks.csv', decimal=',', sep=';')

X = df[['Economy', 'Value', 'Sport.', 'Safety', 'Price']]
#list to get the mark
y = [i for i in range(len(df['Mark']))]
yy = df['Design']

regr = linear_model.LinearRegression()
regr.fit(X, yy)

print(f'Skuteczność: {regr.score(X, yy)}')               #coefficient of determination of prediction (R^2)
print(f'Współczynniki: {regr.coef_}')
print(f'Współczynnik wolny (B0): {regr.intercept_}')

predicted = regr.predict([[2.2, 3.3, 2.9, 1.8, 3.2]])
rounded_pred = round(int(predicted))
#print(f'Przewidywwana wartość: {predicted} {df["Mark"][rounded_pred]}')
print(f'Przewidywwana wartość: {predicted}')