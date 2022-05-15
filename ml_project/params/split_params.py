from dataclasses import dataclass, field
from typing import List


@dataclass()
class SplittingParams:
    stratify: List[str]
    val_size: float = field(default=0.2)
    random_state: int = field(default=42)
