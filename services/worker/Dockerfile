FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY libs/core /app/libs/core
COPY libs/tools /app/libs/tools
COPY schemas /app/schemas
COPY services/worker /app/services/worker
WORKDIR /app/services/worker
RUN pip install --no-cache-dir -e /app/libs/core -e .
CMD ["python", "-m", "app.main"]
