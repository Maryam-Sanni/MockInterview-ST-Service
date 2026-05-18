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
# DOWNLOAD WAV2LIP MODEL
# ==============================

RUN mkdir -p checkpoints

RUN wget -O checkpoints/wav2lip_gan.pth \
    "https://huggingface.co/camenduru/Wav2Lip/resolve/main/wav2lip_gan.pth"

# ==============================
# RUN SERVERLESS HANDLER
# ==============================

CMD ["python", "-u", "handler.py"]