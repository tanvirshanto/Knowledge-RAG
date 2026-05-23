FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only PyTorch first (avoids CUDA wheels and version skew with transformers)
RUN pip install --upgrade pip \
    && pip install "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY main.py .
COPY app ./app

RUN mkdir -p tmp_uploads

EXPOSE 8000



# Use 1 worker: job status is in-memory (see README)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
