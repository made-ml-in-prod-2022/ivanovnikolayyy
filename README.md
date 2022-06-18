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

```
docker build -t <docker_id>/<image_name>:<tag> ./
```

### predict

```
docker run -p 8000:8000 <docker_id>/<image_name>:<tag>
```

after application startup complete, in separate tab run

```
poetry run python ml_project/make_request.py
```
