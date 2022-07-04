from dataclasses import dataclass

import yaml
from marshmallow_dataclass import class_schema

from .classifier_params import ClassifierParams
from .feature_params import FeatureParams
from .split_params import SplittingParams


@dataclass()
class TrainingPipelineParams:
    dataset_path: str
    output_model_path: str
    metric_path: str
    feature_params: FeatureParams
    classifier_params: ClassifierParams
    splitting_params: SplittingParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
