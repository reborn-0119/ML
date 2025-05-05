import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
diabetes = load_diabetes()
X = diabetes.data[:, np.newaxis, 2] # Using only the BMI feature
y = diabetes.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
slr = LinearRegression()
slr.fit(X_train, y_train)

# Predict
y_pred = slr.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2) Score: {r2:.4f}")
print(f"Coefficient: {slr.coef_[0]:.4f}")
print(f"Intercept: {slr.intercept_:.4f}")

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='black', label='Actual data')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Regression line')
plt.xlabel("BMI Feature")
plt.ylabel("Diabetes Progression")
plt.title("Simple Linear Regression Fit")
plt.legend()
plt.grid(True)
plt.show()