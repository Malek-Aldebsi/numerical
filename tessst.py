import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt

# Sample data with multiple dimensions
# X_train = np.array([[0.5, 2], [1, 3], [1.5, 4], [2, 5], [2.5, 6]])
# y_train = np.array([5.38, 4.69, 10.02, 9.04, 14.06])
X_train = np.array([[6, 9], [7, 7], [8, 0], [5, 8], [3, 5], [2, 6], [1, 4], [2, 3], [5, 2], [8, 10]])
y_train = np.array([9, 7, 6, 4, 3, 2, 5, 6, 8, 1])

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)

# Create polynomial features for multiple dimensions
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
# Scatter plot for multiple dimensions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, label='Train set points')

# Generate a grid for 3D plotting
xx, yy = np.meshgrid(np.linspace(min(X_train[:, 0]), max(X_train[:, 0]), 100),
                     np.linspace(min(X_train[:, 1]), max(X_train[:, 1]), 100))
X_grid = np.c_[xx.ravel(), yy.ravel()]
X_grid_norm = scaler.transform(X_grid)
X_grid_poly = poly_features.transform(X_grid_norm)
y_grid_pred = sgdr.predict(X_grid_poly)

# 3D surface plot for the polynomial regression
ax.plot_surface(xx, yy, y_grid_pred.reshape(xx.shape), cmap='viridis', alpha=0.8,
                label='Polynomial curve (degree {})'.format(degree))

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y_train')
plt.title('Train set points and Polynomial curve')
# plt.legend()
plt.grid(True)
plt.show()
