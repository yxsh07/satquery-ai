FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip git \
    gdal-bin libgdal-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY inference/requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY inference/ .
EXPOSE 8001
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
