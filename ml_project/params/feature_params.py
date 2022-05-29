from dataclasses import dataclass, field
from typing import List, Optional

DEFAULT_NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
DEFAULT_CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


@dataclass()
class FeatureParams:
    categorical_features: List[str] = field(
        default_factory=lambda: DEFAULT_CATEGORICAL_FEATURES
    )
    numerical_features: List[str] = field(
        default_factory=lambda: DEFAULT_NUMERICAL_FEATURES
    )
    target_col: Optional[str] = field(default="condition")
