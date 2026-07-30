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

# Run behind a reverse proxy, correctly.
#
# `sh -c` rather than the exec form because both values have to come from the
# environment; `exec` keeps uvicorn as PID 1 so it still receives SIGTERM and
# shuts down cleanly.
#
# --proxy-headers is on by default in uvicorn, but it is useless on its own:
#   `--forwarded-allow-ips` defaults to 127.0.0.1, and Railway's proxy reaches
#   the container from an internal address that is not loopback. Every
#   X-Forwarded-* header was therefore discarded, which meant:
#     * `request.client.host` was the proxy's address, so the whole platform
#       shared ONE rate-limit bucket — the first five sign-in attempts from
#       anybody locked out every other user for the rest of the minute;
#     * `request.url.scheme` was http behind TLS termination.
#
#   FORWARDED_ALLOW_IPS defaults to `*` here because the container is only
#   reachable through the platform's ingress. Set it to the proxy's address
#   range on any deployment where the container is directly addressable —
#   trusting the header from an arbitrary client lets that client choose the
#   IP its rate limit is counted against.
#
# PORT is injected by Railway; the fallback keeps `docker run` and compose
# working unchanged.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips \"${FORWARDED_ALLOW_IPS:-*}\""]
