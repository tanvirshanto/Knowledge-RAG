FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    DOCLING_ARTIFACTS_PATH=/app/.cache/docling/models \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

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

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Pre-download RapidOCR models (only ~15MB, baked into image)
RUN python -c "import os, urllib.request; d='/usr/local/lib/python3.11/site-packages/rapidocr/models'; os.makedirs(d, exist_ok=True); [urllib.request.urlretrieve(u, os.path.join(d, u.split('/')[-1])) for u in ['https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0/onnx/PP-OCRv4/det/ch_PP-OCRv4_det_mobile.onnx', 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx', 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0/onnx/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile.onnx']]; print('Models cached successfully!')"

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
COPY vertex_rag ./vertex_rag

RUN mkdir -p tmp_uploads

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
