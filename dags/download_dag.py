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
    "download",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=pendulum.now(tz="Europe/Moscow").add(days=-3),
    tags=["airflow"],
) as dag:

    download = DockerOperator(
        image="airflow-download",
        command="--output-dir /data/raw/{{ ds }}",
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

    download
