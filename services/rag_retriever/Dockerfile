FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app
COPY libs/core /app/libs/core
COPY services/rag_retriever /app/services/rag_retriever
WORKDIR /app/services/rag_retriever
RUN pip install --no-cache-dir -e /app/libs/core -e .
EXPOSE 8086
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8086"]
