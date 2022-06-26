# ivanovnikolayyy
Homework for 'Machine Learning in Production' course

## Local

### install

```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
poetry install
```

### train
```
poetry run python ml_project/run.py train configs/train_config.yaml
```

### predict

```
poetry run python ml_project/run.py predict models/model.pkl data/train.csv predicts.npy
```

### run app

```
PATH_TO_MODEL="models/model.pkl" poetry run python ml_project/app.py
```

### tests

```
poetry run pytest tests/.
```

## Docker

### install

pull latest build

```
docker pull ivanovnikolayyy/made_ml_in_prod:latest
```

or build your own docker image

```
docker build -t <docker_id>/<name>:<tag> .
```

### predict

run app in docker container

```
docker run -p 8000:8000 <docker_id>/<name>:<tag>
```

after application startup complete, in separate tab run

```
python ml_project/make_request.py
```

### airflow

```
docker compose down -v
docker compose up --build
```

