FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app
COPY libs/core /app/libs/core
COPY services/critic /app/services/critic
WORKDIR /app/services/critic
RUN pip install --no-cache-dir -e /app/libs/core -e .
CMD ["python", "-m", "app.main"]
