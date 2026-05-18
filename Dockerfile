FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

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
    curl \
    && rm -rf /var/lib/apt/lists/*

# ==============================
# COPY PROJECT
# ==============================

COPY . /app

# ==============================
# PYTHON DEPENDENCIES
# ==============================

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ==============================
# WAV2LIP CHECKPOINT
# ==============================

RUN mkdir -p checkpoints && \
    curl -L "https://huggingface.co/rippertnt/wav2lip/resolve/main/checkpoints/wav2lip_gan.pth" \
    -o checkpoints/wav2lip_gan.pth

# ==============================
# START HANDLER
# ==============================

CMD ["python", "-u", "handler.py"]