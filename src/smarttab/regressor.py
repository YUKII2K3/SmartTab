"""SmartTabRegressor class.

This module provides the SmartTabRegressor class for regression tasks.

Example:
    from smarttab import SmartTabRegressor

    model = SmartTabRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from smarttab.base import (
    _initialize_model_variables_helper,
    check_cpu_warning,
    create_inference_engine,
    determine_precision,
    get_preprocessed_datasets_helper,
    initialize_smarttab_model,
)
from smarttab.inference import InferenceEngine, InferenceEngineBatchedNoPreprocessing
from smarttab.model.bar_distribution import FullSupportBarDistribution
from smarttab.model.loading import (
    load_fitted_smarttab_model,
    save_fitted_smarttab_model,
)
from smarttab.preprocessing import (
    BaseDatasetConfig,
    RegressorDatasetConfig,
    DatasetCollectionWithPreprocessing,
    fit_preprocessing,
    get_preprocessing_config,
)
from smarttab.utils import (
    infer_device_and_type,
    infer_fp16_inference_mode,
    infer_random_state,
    split_large_data,
    update_encoder_params,
)

from smarttab.config import ModelInterfaceConfig
from smarttab.constants import XType, YType
from smarttab.model.config import ModelConfig

if __name__ == "__main__":
    import sklearn.datasets
    from smarttab import SmartTabRegressor

    model = SmartTabRegressor()
    X, y = sklearn.datasets.make_regression(n_samples=50, n_features=10)

    model.fit(X, y)
    predictions = model.predict(X)
