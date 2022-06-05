import os
from pathlib import Path
from typing import List

from ml_project.params import (
    ClassifierParams,
    FeatureParams,
    SplittingParams,
    TrainingPipelineParams,
)
from ml_project.run import train_pipeline


def test_train_e2e(
    tmpdir: Path,
    dataset_path: str,
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
):
    expected_output_model_path = os.path.join(tmpdir, "model.pkl")
    expected_metric_path = os.path.join(tmpdir, "metrics.pkl")
    params = TrainingPipelineParams(
        input_data_path=dataset_path,
        output_model_path=expected_output_model_path,
        metric_path=expected_metric_path,
        splitting_params=SplittingParams(
            val_size=0.2, random_state=42, stratify=["condition"]
        ),
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_col=target_col,
        ),
        classifier_params=ClassifierParams(model_type="LogisticRegression"),
    )
    real_model_path, metrics = train_pipeline(params)
    assert metrics["f1_score"] > 0
    assert os.path.exists(real_model_path)
    assert os.path.exists(params.metric_path)
