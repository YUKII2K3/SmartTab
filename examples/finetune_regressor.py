#  Copyright (c) yuktheshwar 2025.
"""Example of fine-tuning SmartTab for regression.

This example demonstrates how to fine-tune SmartTab on a custom regression dataset.
"""

import numpy as np
import torch
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from smarttab import SmartTabRegressor
from smarttab.model.loading import save_fitted_smarttab_model

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic regression data
X, y = make_regression(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    noise=0.1,
    random_state=42,
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SmartTab regressor with fine-tuning enabled
regressor = SmartTabRegressor(
    device="cpu",  # Change to "cuda" if you have a GPU
    fit_mode="finetune",  # Enable fine-tuning
    finetune_epochs=10,  # Number of fine-tuning epochs
    finetune_lr=1e-4,  # Learning rate for fine-tuning
)

# Fine-tune the model
print("Starting fine-tuning...")
regressor.fit(X_train, y_train)

# Evaluate the fine-tuned model
train_score = regressor.score(X_train, y_train)
test_score = regressor.score(X_test, y_test)

print(f"Training R² score: {train_score:.4f}")
print(f"Test R² score: {test_score:.4f}")

# Save the fine-tuned model
save_fitted_smarttab_model(regressor, "fine_tuned_regressor.smarttab_fit")
print("Fine-tuned model saved to 'fine_tuned_regressor.smarttab_fit'")

# Make predictions with the fine-tuned model
predictions = regressor.predict(X_test)

print(f"Predictions shape: {predictions.shape}")
print(f"First 5 predictions: {predictions[:5]}")
print(f"First 5 actual values: {y_test[:5]}")
