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
    "train",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=pendulum.now(tz="Europe/Moscow").add(days=-3),
    tags=["airflow"],
) as dag:

    split = DockerOperator(
        image="airflow-split",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/split/{{ ds }}",
        task_id="docker-airflow-split",
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

    train = DockerOperator(
        image="airflow-train",
        command="--train-dataset-dir /data/split/{{ ds }}/train --output-dir /data/models/{{ ds }}",
        task_id="docker-airflow-train",
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

    validate = DockerOperator(
        image="airflow-validate",
        command="--test-dataset-dir /data/split/{{ ds }}/test "
        "--model-path /data/models/{{ ds }} "
        "--output-dir /data/metrics/{{ ds }}",
        task_id="docker-airflow-validate",
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

    split >> train >> validate
