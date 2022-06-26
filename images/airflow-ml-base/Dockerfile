FROM python:3.8
ENV POETRY_VERSION = 1.1.11

RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /build/

COPY poetry.lock pyproject.toml /build/

RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --no-dev

WORKDIR /code/

COPY . /code/
