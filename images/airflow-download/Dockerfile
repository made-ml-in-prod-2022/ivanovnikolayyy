FROM airflow-ml-base:latest

COPY download.py fake.py reference_data.csv ./

ENTRYPOINT ["python", "download.py"]
