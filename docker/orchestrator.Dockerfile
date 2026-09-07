FROM python:3.11-slim
WORKDIR /app
COPY orchestrator/requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
COPY orchestrator/ /app/orchestrator/
EXPOSE 8000
CMD ["uvicorn", "orchestrator.main:app", "--host", "0.0.0.0", "--port", "8000"]
