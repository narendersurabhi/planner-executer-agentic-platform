FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app
COPY libs/core /app/libs/core
COPY libs/framework /app/libs/framework
COPY libs/tools /app/libs/tools
COPY config /app/config
COPY schemas /app/schemas
COPY services/api /app/services/api
WORKDIR /app/services/api
RUN pip install --no-cache-dir -e /app/libs/core -e .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
