"""Example of saving and loading a fitted SmartTab model."""

from __future__ import annotations

# Copyright (c) Prior Labs GmbH 2025.
from pathlib import Path

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from smarttab import SmartTabRegressor
from smarttab.model.loading import (
    load_fitted_smarttab_model,
    save_fitted_smarttab_model,
)

# Train a regressor on GPU
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
reg = SmartTabRegressor(device="cuda")
reg.fit(X_train, y_train)

# Save the fitted estimator
save_fitted_smarttab_model(reg, Path("trained_reg.smarttab_fit"))

# Load on CPU for inference
reg_cpu = load_fitted_smarttab_model(Path("trained_reg.smarttab_fit"), device="cpu")
print(reg_cpu.predict(X_test)[:5])
