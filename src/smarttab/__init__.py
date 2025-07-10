from importlib.metadata import version

from smarttab.classifier import SmartTabClassifier
from smarttab.misc.debug_versions import display_debug_info
from smarttab.model.loading import (
    load_fitted_smarttab_model,
    save_fitted_smarttab_model,
)
from smarttab.regressor import SmartTabRegressor

try:
    __version__ = version(__name__)
except ImportError:
    __version__ = "unknown"

__all__ = [
    "SmartTabClassifier",
    "SmartTabRegressor",
    "__version__",
    "display_debug_info",
    "load_fitted_smarttab_model",
    "save_fitted_smarttab_model",
]
