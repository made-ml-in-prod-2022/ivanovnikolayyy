import json
import logging
import sys

import mlflow
import numpy as np
import typer
from sklearn.pipeline import Pipeline

from ml_project.classifiers import (
    evaluate_classifier,
    load_model,
    make_classifier,
    save_model,
)
from ml_project.data import read_data, split_train_test_data
from ml_project.features import make_transformer
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
        data, targets = read_data(training_pipeline_params.dataset_path)
        logger.info(f"data.shape is {data.shape}")

        data_train, data_test, targets_train, targets_test = split_train_test_data(
            data, targets, training_pipeline_params.splitting_params
        )
        logger.info(f"data_train.shape is {data_train.shape}")
        logger.info(f"data_test.shape is {data_test.shape}")

        transforms = make_transformer(training_pipeline_params.feature_params)
        clf = make_classifier(training_pipeline_params.classifier_params)
        model = Pipeline([("transforms", transforms), ("clf", clf)])

        logger.info("fitting model")
        model.fit(data_train, targets_train)

        logger.info("saving model")
        path_to_model = save_model(model, training_pipeline_params.output_model_path)

        logger.info("evaluating test metrics")
        predicts = model.predict(data_test)
        metrics = evaluate_classifier(targets_test, predicts)
        mlflow.log_metrics(metrics)

        with open(training_pipeline_params.metric_path, "w") as metric_file:
            json.dump(metrics, metric_file)
        logger.info(f"metrics is {metrics}")

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
