# Matches the API image's Python version so the worker and the web process
# never run on different interpreters.
FROM python:3.12-slim

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The worker does NOT run migrations — the API container owns schema changes, so
# two services can never race to apply the same revision.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /workspace/uploads \
    && chown -R appuser:appuser /workspace
USER appuser

CMD ["celery", "-A", "app.worker.celery_app", "worker", "--loglevel=info"]
