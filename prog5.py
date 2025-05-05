import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load data
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("--- Lasso Regression ---")
print(f"Mean Squared Error (MSE): {mse_lasso:.4f}")
print(f"R-squared (R2) Score: {r2_lasso:.4f}")
print(f"Coefficients: {lasso.coef_}")
print(f"Intercept: {lasso.intercept_:.4f}")

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("\n--- Ridge Regression ---")
print(f"Mean Squared Error (MSE): {mse_ridge:.4f}")
print(f"R-squared (R2) Score: {r2_ridge:.4f}")
print(f"Coefficients: {ridge.coef_}")
print(f"Intercept: {ridge.intercept_:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_ridge, alpha=0.7, edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) 
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values (Ridge)")
plt.title("Ridge Regression: Actual vs. Predicted")
plt.grid(True)
plt.show()
