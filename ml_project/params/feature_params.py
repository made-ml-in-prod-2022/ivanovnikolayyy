from dataclasses import dataclass, field
from typing import List


@dataclass()
class FeatureParams:
    categorical_features: List[str] = field(default_factory=lambda: [])
    numerical_features: List[str] = field(default_factory=lambda: [])
