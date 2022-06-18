import pytest
from fastapi.testclient import TestClient

from ml_project.app import app


@pytest.fixture()
def request_json():
    return {
        "data": [60, 1, 2, 140, 185, 0, 2, 155, 0, 3.0, 1, 0, 0],
        "features": [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ],
    }


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()


def test_predict(request_json):
    with TestClient(app) as client:
        response = client.get("/predict/", json=request_json)
        assert response.status_code == 200
        assert response.json()[0]["condition"] == 0
