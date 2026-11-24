FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATABASE_PATH=/data/app.db \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend/ /app/backend/
COPY static/ /app/static/
COPY examples/ /app/examples/
COPY .env.example /app/.env.example

RUN mkdir -p /data

EXPOSE 8000

# IMPORTANT: Run with --workers 1 because WebSocket connections, session locks,
# and in-flight AI task handles are coordinated in-process alongside SQLite WAL.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
