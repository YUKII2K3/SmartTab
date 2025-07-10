"""
Example script for regression with SmartTab.
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from smarttab import SmartTabRegressor

# Generate synthetic regression data
X, y = make_regression(
    n_samples=1000, n_features=20, n_informative=10, noise=0.1, random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train SmartTab regressor
regressor = SmartTabRegressor(device="cpu")
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
