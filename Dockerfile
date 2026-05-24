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

RUN pip install --upgrade pip \
    && pip install "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY main.py .
COPY seed.py .
COPY app ./app
COPY api ./api
COPY auth ./auth
COPY middleware ./middleware
COPY repositories ./repositories
COPY services ./services
COPY workers ./workers
COPY schemas ./schemas
COPY utils ./utils

RUN mkdir -p tmp_uploads

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
