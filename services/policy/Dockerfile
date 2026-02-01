FROM python:3.11-slim
WORKDIR /app
COPY libs/core /app/libs/core
COPY services/policy /app/services/policy
WORKDIR /app/services/policy
RUN pip install --no-cache-dir -e /app/libs/core -e .
CMD ["python", "-m", "app.main"]
