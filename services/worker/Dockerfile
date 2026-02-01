FROM python:3.11-slim
WORKDIR /app
COPY libs/core /app/libs/core
COPY services/worker /app/services/worker
WORKDIR /app/services/worker
RUN pip install --no-cache-dir -e /app/libs/core -e .
CMD ["python", "-m", "app.main"]
