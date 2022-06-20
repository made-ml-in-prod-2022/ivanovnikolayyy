FROM python:3.8
ENV POETRY_VERSION = 1.1.11

RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /code/
COPY poetry.lock pyproject.toml /code/

RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

COPY images/airflow-validate/model.pkl ml_project/app.py /code/

ENV PATH_TO_MODEL="/code/model.pkl"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
