FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app
COPY libs/core /app/libs/core
COPY services/tailor /app/services/tailor
WORKDIR /app/services/tailor
RUN pip install --no-cache-dir -e /app/libs/core -e .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
