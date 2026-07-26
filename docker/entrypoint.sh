#!/usr/bin/env sh
#
# Container entrypoint.
#
# Applies database migrations before serving traffic. Without this a fresh
# deployment starts against an empty database and every request fails with
# UndefinedTableError — the schema must never depend on someone remembering to
# run Alembic by hand.
#
set -eu

echo "[entrypoint] applying database migrations..."
alembic upgrade head
echo "[entrypoint] migrations applied; database is at head"

echo "[entrypoint] starting: $*"
exec "$@"
