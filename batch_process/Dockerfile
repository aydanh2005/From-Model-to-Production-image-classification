FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir pandas requests psycopg2-binary schedule

COPY batch_process/batch.py .
COPY batch_process/scheduler.py .

RUN mkdir -p upload

CMD ["python", "scheduler.py"]
