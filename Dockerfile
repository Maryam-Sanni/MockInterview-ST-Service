FROM runpod/pytorch:3.10-2.0.1-120-devel

WORKDIR /app

# ==============================
# SYSTEM DEPENDENCIES
# ==============================
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ==============================
# COPY REQUIREMENTS FIRST (CACHE OPTIMIZED)
# ==============================
COPY requirements.txt /app/

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ==============================
# COPY PROJECT (BEFORE MODEL DOWNLOAD)
# ==============================
COPY . /app

# ==============================
# DOWNLOAD WAV2LIP CHECKPOINT (AFTER COPY)
# ==============================
RUN mkdir -p /app/Wav2Lip/checkpoints && \
    curl -L --retry 3 --fail \
    -o /app/Wav2Lip/checkpoints/wav2lip_gan.pth \
    https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth

# ==============================
# START
# ==============================
CMD ["python", "-u", "handler.py"]