"""EuroTender: CPV classification for European public procurement notices."""

from .model import CpvModel
from .predict import CpvPrediction, load_model, predict_cpv

__version__ = "0.1.0"

__all__ = ["CpvModel", "CpvPrediction", "__version__", "load_model", "predict_cpv"]
