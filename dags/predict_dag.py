from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

MODEL_DIR = Variable.get("MODEL_DIR")
INPUT_DIR = "/data/raw/{{ ds }}"
OUTPUT_DIR = "/data/predicts/{{ ds }}"

default_args = {
    "owner": "ivanovnikolayyy",
    "email": ["kolyaslaplace@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "predict",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=pendulum.now(tz="Europe/Moscow").add(days=-3),
    tags=["airflow"],
) as dag:

    predict = DockerOperator(
        image="airflow-predict",
        command=f"--input-dir {INPUT_DIR} --model-dir {MODEL_DIR} --output-dir {OUTPUT_DIR}",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[
            Mount(
                source="/Users/nikolai.ivanov/Documents/made/ivanovnikolayyy/data/",
                target="/data",
                type="bind",
            )
        ],
    )

    predict
