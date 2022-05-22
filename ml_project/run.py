import json
import logging
import sys

import mlflow
import numpy as np
import typer
from sklearn.pipeline import Pipeline

from ml_project.data import read_data, split_train_val_data
from ml_project.features import build_transformer, extract_features_and_target
from ml_project.models import (
    build_classifier,
    evaluate_classifier,
    load_model,
    save_model,
)
from ml_project.params.pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)

TRAIN_CONFIG_PATH = "configs/train_config.yaml"

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

app = typer.Typer()


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    logger.info("start mlflow experiment")
    logger.info("mlflow experiment started")
    with mlflow.start_run():
        logger.info(f"start train pipeline with params {training_pipeline_params}")
        data = read_data(training_pipeline_params.input_data_path)
        logger.info(f"data.shape is {data.shape}")
        train_df, val_df = split_train_val_data(data, training_pipeline_params.splitting_params)
        logger.info(f"train_df.shape is {train_df.shape}")
        logger.info(f"val_df.shape is {val_df.shape}")

        transforms = build_transformer(training_pipeline_params.feature_params)
        clf = build_classifier(training_pipeline_params.classifier_params)
        model = Pipeline([("transforms", transforms), ("clf", clf)])

        train_features, train_target = extract_features_and_target(train_df, training_pipeline_params.feature_params)
        logger.info(f"train_features.shape is {train_features.shape}")

        model.fit(train_features, train_target)

        val_features, val_target = extract_features_and_target(val_df, training_pipeline_params.feature_params)
        logger.info(f"val_features.shape is {val_features.shape}")

        predicts = model.predict(val_features)
        metrics = evaluate_classifier(predicts, val_target)
        mlflow.log_metrics(metrics)

        with open(training_pipeline_params.metric_path, "w") as metric_file:
            json.dump(metrics, metric_file)
        logger.info(f"metrics is {metrics}")

        path_to_model = save_model(model, training_pipeline_params.output_model_path)

        return path_to_model, metrics


@app.command()
def train(config_path: str):
    params = read_training_pipeline_params(config_path)
    train_pipeline(params)


@app.command()
def predict(model_path: str, data_path: str, save_path: str):
    model = load_model(model_path)
    data = read_data(data_path)
    predicts = model.predict(data)
    np.save(save_path, predicts)


if __name__ == "__main__":
    app()
