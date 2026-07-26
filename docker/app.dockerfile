# Python 3.12 to match the development environment and the resolved dependency
# set. This previously pinned 3.11, so container behaviour could diverge from
# everything that was actually tested.
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

# Line endings are normalised because this repository is developed on Windows;
# a CRLF shebang makes the entrypoint fail with "no such file or directory".
RUN sed -i 's/\r$//' docker/entrypoint.sh \
    && chmod +x docker/entrypoint.sh

# Run as a non-root user. `uploads/` is created here because it is excluded from
# the build context (see .dockerignore) and mounted as a volume at runtime.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /workspace/uploads \
    && chown -R appuser:appuser /workspace
USER appuser

EXPOSE 8000

# Migrations run before the server starts, so a fresh deployment cannot come up
# against an empty schema.
ENTRYPOINT ["./docker/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
