import numpy as np


# 1. Implement linear regression using NumPy
def linear_regression(X, y):
  # Add a column of ones to X for the bias term (intercept)
  X_b = np.c_[np.ones((X.shape[0], 1)), X]

  # Calculate the parameters using the Normal Equation: θ = (X^T X)^(-1) X^T y
  theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

  return theta


# 2. Create a simple dataset
np.random.seed(42)  # For reproducibility
X = 2 * np.random.rand(100, 1)  # 100 random data points
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relation with some noise

# 3. Train the model
theta = linear_regression(X, y)
print(f"Calculated model parameters (theta):\n{theta}")


# 4. Evaluate the model's performance using Mean Squared Error (MSE)
def mean_squared_error(y_true, y_pred):
  return ((y_true - y_pred) ** 2).mean()


# Predict the values of y using the model
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add the bias term
y_pred = X_b.dot(theta)

# Calculate the MSE
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")
