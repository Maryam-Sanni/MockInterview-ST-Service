FROM runpod/pytorch:3.10-2.0.1-120-devel

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# =========================
# SYSTEM DEPENDENCIES
# =========================
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
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
# COPY APP CODE
# =========================
COPY . /app

# =========================
# PRE-DOWNLOAD MODELS (CRITICAL FOR SPEED)
# =========================

# Wav2Lip checkpoint
RUN mkdir -p /app/Wav2Lip/checkpoints && \
    wget -O /app/Wav2Lip/checkpoints/wav2lip_gan.pth \
    https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth

# Face detector
RUN mkdir -p /app/Wav2Lip/face_detection/detection/sfd && \
    wget -O /app/Wav2Lip/face_detection/detection/sfd/s3fd.pth \
    https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth

# Optional: pre-cache base video placeholder (replace with your own)
# COPY inputs/video.mp4 /app/base.mp4

CMD ["python", "-u", "handler.py"]