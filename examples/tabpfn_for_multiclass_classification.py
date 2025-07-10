"""
Example script for multiclass classification with SmartTab.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from smarttab import SmartTabClassifier

# Generate synthetic multiclass classification data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=3,
    random_state=42,
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train SmartTab classifier
classifier = SmartTabClassifier(device="cpu")
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)
y_pred_proba = classifier.predict_proba(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Prediction probabilities shape: {y_pred_proba.shape}")
