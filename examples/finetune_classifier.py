#  Copyright (c) yuktheshwar 2025.
"""Example of fine-tuning SmartTab for classification.

This example demonstrates how to fine-tune SmartTab on a custom classification dataset.
"""

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from smarttab import SmartTabClassifier
from smarttab.model.loading import save_fitted_smarttab_model

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic classification data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    random_state=42,
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SmartTab classifier with fine-tuning enabled
classifier = SmartTabClassifier(
    device="cpu",  # Change to "cuda" if you have a GPU
    fit_mode="finetune",  # Enable fine-tuning
    finetune_epochs=10,  # Number of fine-tuning epochs
    finetune_lr=1e-4,  # Learning rate for fine-tuning
)

# Fine-tune the model
print("Starting fine-tuning...")
classifier.fit(X_train, y_train)

# Evaluate the fine-tuned model
train_score = classifier.score(X_train, y_train)
test_score = classifier.score(X_test, y_test)

print(f"Training accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")

# Save the fine-tuned model
save_fitted_smarttab_model(classifier, "fine_tuned_classifier.smarttab_fit")
print("Fine-tuned model saved to 'fine_tuned_classifier.smarttab_fit'")

# Make predictions with the fine-tuned model
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)

print(f"Predictions shape: {predictions.shape}")
print(f"Probabilities shape: {probabilities.shape}")
print(f"First 5 predictions: {predictions[:5]}")
print(f"First 5 probabilities: {probabilities[:5]}")
