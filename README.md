# ivanovnikolayyy
Homework for 'Machine Learning in Production' course

# Install

```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
poetry install
poetry shell
```

# Run

## 1. train
```
python ml_project/run.py train configs/train_config.yaml
```

## 2. predict
```
python ml_project/run.py predict models/model.pkl data/train.csv predicts.npy
```

## 3. tests

```
pytest tests/.
```