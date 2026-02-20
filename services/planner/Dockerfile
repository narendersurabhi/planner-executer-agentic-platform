FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app
COPY libs/core /app/libs/core
COPY libs/tools /app/libs/tools
COPY services/planner /app/services/planner
WORKDIR /app/services/planner
RUN pip install --no-cache-dir -e /app/libs/core -e .
CMD ["python", "-m", "app.main"]
