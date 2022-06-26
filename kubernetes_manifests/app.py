import logging
import os
import pickle
import time
from typing import List, Optional, Union

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

DEFAULT_PATH_TO_MODEL = "models/model.pkl"

logger = logging.getLogger(__name__)


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


class HeartDiseaseModel(BaseModel):
    data: List[Union[float, str, None]]
    features: List[str]


class HeartDiseaseResponse(BaseModel):
    id: str
    condition: float


model: Optional[Pipeline] = None


def make_predict(
    data: List, features: List[str], model: Pipeline
) -> List[HeartDiseaseResponse]:
    data = pd.DataFrame([data], columns=features)
    preds = model.predict(data)

    return [
        HeartDiseaseResponse(id=id_, condition=pred) for id_, pred in enumerate(preds)
    ]


app = FastAPI()


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("PATH_TO_MODEL", DEFAULT_PATH_TO_MODEL)
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    model = load_object(model_path)


@app.get("/health")
def health() -> bool:
    start_time = time.time()
    time.sleep(30)
    if time.time() - start_time > 20:
        raise RuntimeError("Not responding")

    return model is not None


@app.get("/predict/", response_model=List[HeartDiseaseResponse])
def predict(request: HeartDiseaseModel):
    return make_predict(request.data, request.features, model)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
