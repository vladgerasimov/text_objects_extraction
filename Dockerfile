FROM python:3.12
WORKDIR /app

RUN pip install poetry==1.6.1

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-cache --no-root --without dev \
    && python -m spacy download en_core_web_sm

COPY data.dvc .
COPY configs configs/
COPY src src/

CMD uvicorn src.fastapi_app:app --reload --host 0.0.0.0 --port 8000
