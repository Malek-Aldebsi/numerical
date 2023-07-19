import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt

X_train = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y_train = np.array([3, 2, 0.5, 1.5, 4])
# X_train = np.array([0.5, 1, 1.5, 2, 2.5]).reshape(-1, 1)
# y_train = np.array([5.38, 4.69, 10.02, 9.04, 14.06])

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)

# Create polynomial features
degree = 3  # Set the degree of the polynomial (you can change it as needed)
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X_norm)

sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_poly, y_train)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")

# Predict using the polynomial features
y_pred_sgd = sgdr.predict(X_poly)

####################
# Plot the train set points
plt.scatter(X_train, y_train, label='Train set points')

# Plot the polynomial curve
x_range = np.linspace(min(X_train), max(X_train), 100).reshape(-1, 1)
x_range_norm = scaler.transform(x_range)
x_range_poly = poly_features.transform(x_range_norm)
y_pred_range = sgdr.predict(x_range_poly)
plt.plot(x_range, y_pred_range, color='red', label='Polynomial curve (degree {})'.format(degree))

plt.xlabel('X_train')
plt.ylabel('y_train')
plt.title('Train set points and Polynomial curve')
plt.legend()
plt.grid(True)
plt.show()
