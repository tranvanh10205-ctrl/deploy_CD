
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV TS_TCC_ROOT=/app
ENV CHECKPOINT_PATH=/app/experiments_logs/SpeechOcean762_TS-TCC/linear_bn_fixed/train_linear_seed_0/saved_models/ckp_best.pt
ENV DEVICE=cpu
ENV WHISPER_MODEL=tiny

WORKDIR /app

# FFmpeg dùng để giải mã WebM, MP3, WAV...
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Cài PyTorch bản CPU trước
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Cài các dependency của ứng dụng
COPY speakify_local_web/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy repository: mã nguồn mô hình, checkpoint và ứng dụng
COPY . /app

EXPOSE 10000

CMD ["sh", "-c", "uvicorn speakify_local_web.app:app --host 0.0.0.0 --port ${PORT:-10000}"]
