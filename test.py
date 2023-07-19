import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# X_train = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
# y_train = np.array([3, 2, 0.5, 1.5, 4])
X_train = np.array([0.5, 1, 1.5, 2, 2.5]).reshape(-1, 1)
y_train = np.array([5.38, 4.69, 10.02, 9.04, 14.06])

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)

sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")

y_pred_sgd = sgdr.predict(X_norm)

print(f"Prediction on training set:\n{y_pred_sgd[:4]}" )
print(f"Target values \n{y_train[:4]}")

####################
plt.scatter(X_train, y_train, label='Train set points')

# Plot the linear curve
x_range = np.linspace(min(X_train), max(X_train), 100).reshape(-1, 1)
x_range_norm = scaler.transform(x_range)
y_pred_range = sgdr.predict(x_range_norm)
plt.plot(x_range, y_pred_range, color='red', label='Linear curve')

plt.xlabel('X_train')
plt.ylabel('y_train')
plt.title('Train set points and Linear curve')
plt.legend()
plt.grid(True)
plt.show()