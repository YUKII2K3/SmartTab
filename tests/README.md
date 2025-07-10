# SmartTab Tests

This directory contains tests for the SmartTab project.

## Test Files

- `test_classifier_interface.py`: Tests for the SmartTabClassifier interface
- `test_regressor_interface.py`: Tests for the SmartTabRegressor interface
- `test_preprocessing.py`: Tests for preprocessing functionality
- `test_finetuning_classifier.py`: Tests for classifier fine-tuning

## Consistency Tests

The consistency tests verify SmartTab models produce consistent predictions across code changes, ensuring:

1. Model predictions are deterministic
2. Model behavior is consistent across different environments
3. Model performance is maintained across updates

## Running Tests

1. Creates a SmartTab model with fixed settings
2. Trains on a small dataset
3. Makes predictions and saves results
4. Compares against expected outputs

To run the tests:

```bash
pytest tests/
```