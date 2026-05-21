FROM runpod/pytorch:3.10-2.0.1-120-devel

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/cache
ENV TORCH_HOME=/cache

# =========================
# SYSTEM DEPENDENCIES
# =========================
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# =========================
# PYTHON DEPENDENCIES
# =========================
COPY requirements.txt /app/

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# =========================
# COPY ONLY WHAT YOU NEED
# =========================
COPY handler.py /app/
COPY Wav2Lip /app/Wav2Lip

# =========================
# PRE-DOWNLOAD MODELS (ONCE AT BUILD TIME)
# =========================
RUN mkdir -p /app/Wav2Lip/checkpoints && \
    wget -O /app/Wav2Lip/checkpoints/wav2lip_gan.pth \
    https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth

RUN mkdir -p /app/Wav2Lip/face_detection/detection/sfd && \
    wget -O /app/Wav2Lip/face_detection/detection/sfd/s3fd.pth \
    https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth

CMD ["python", "-u", "handler.py"]