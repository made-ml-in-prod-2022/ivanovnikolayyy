from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

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
        command="--input-dir /data/raw/{{ ds }} --model-dir /data/models/{{ ds }} --output-dir /data/predicts/{{ ds }}",
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
