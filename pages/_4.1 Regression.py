from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

sns.set_theme(color_codes=True)
tips = sns.load_dataset("tips")

data = pd.read_excel(r"C:\Users\user\Desktop\Book1.xlsx")
x = []
column = data.columns.to_list()
# xprint(column)
for index, row in data.iterrows():
    temp_row = []
    for i in column:
        temp_row.append(row[i])
    x.append(temp_row)
print(x)

# x = [  [6, 9,2], [7, 7,3], [8, 0,4], [5, 8,5], [3, 5,6], [2, 6,7], [1, 4,8], [2, 3,9] ,[5,2,1],[8,8,3]]
y = [9, 7, 6, 4, 3, 2, 5, 6]
x, y = np.array(x), np.array(y)
print(x[:, 1])

model = LinearRegression().fit(x, y)
y_pred = model.predict(x)

fig = plt.figure(figsize=(100, 100))
ax = plt.axes(projection='3d')

X, Y = np.meshgrid(x[:, 0], x[:, 1])

surf = ax.plot_surface(x[:, 0], x[:, 1], y_pred.reshape(x.shape), cmap=plt.cm.cividis)

# Set axes label
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)

fig.colorbar(surf, shrink=0.5, aspect=8)

plt.show()

print(y_pred)


def linearRegression(x, y):
    model = LinearRegression()

    model.fit(x, y)

    r_sq = model.score(x, y)
    coaf = model.coef_

    y_pred = model.predict(x)
    St = 0
    Sr = 0
    ave = 0

    for i in range(0, len(y)):
        ave = +y[i]
    ave = ave / len(y)

    for i in range(0, len(y)):
        St = St + (y[i] - ave) ** 2
        Sr = Sr + (y[i] - y_pred[i]) ** 2

    Sd = (St / (len(y) - 1)) ** 0.5
    Syx = (Sr / (len(y) - 2)) ** 0.5

    plt.scatter(x, y, color="red")
    plt.plot(x, y_pred, color="green")
    plt.show()

    return r_sq, model.coef_, model.intercept_, Sr, St


# linearRegression(x,y)


def poylnomial(x, y, degree):
    mymodel = np.poly1d(np.polyfit(x, y, degree))
    myline = np.linspace(1, len(x), 100)

    r2 = r2_score(y, mymodel(x))

    y_pred = []
    St = 0
    Sr = 0
    ave = 0

    for i in range(0, len(y)):
        y_pred.append(mymodel(x[i]))
        ave = +y[i]
    ave = ave / len(y)

    for i in range(0, len(y)):
        St = St + (y[i] - ave) ** 2
        Sr = Sr + (y[i] - y_pred[i]) ** 2

    Sd = (St / (len(y) - 1)) ** 0.5
    Syx = (Sr / (len(y) - 2)) ** 0.5

    plt.scatter(x, y)

    plt.plot(myline, mymodel(myline))

    plt.show()

    print(f"sd= {Sd} ave= {ave} r2={r2} syx= {Syx}")

# poylnomial(x,y,7)