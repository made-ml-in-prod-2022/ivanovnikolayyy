import os
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
    "docker",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=pendulum.datetime(2022, 6, 18, tz="Europe/Moscow"),
) as dag:
    download = DockerOperator(
        image="airflow-download",
        command="--reference-data-path reference_data.csv --output-dir /data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-download",
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

    split = DockerOperator(
        image="airflow-split",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/train/{{ ds }}",
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
        command="--input-dir /data/train/{{ ds }} --output-dir /data/train/{{ ds }}",
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
        command="--input-dir /data/train/{{ ds }} --output-dir /data/metrics/{{ ds }}",
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

    download >> split >> train >> validate
