from dataclasses import dataclass, field


@dataclass()
class ClassifierParams:
    model_type: str = field(default="RandomForestClassifier")
    random_state: int = field(default=42)
    max_iter: int = field(default=100)
