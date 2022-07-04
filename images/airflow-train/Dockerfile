FROM airflow-ml-base:latest

COPY train.py train_config.yaml ./

ENTRYPOINT ["python", "train.py"]
