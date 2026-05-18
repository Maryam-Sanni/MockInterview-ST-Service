FROM runpod/pytorch:3.10-2.0.1-120-devel

# ==============================
# SYSTEM SETUP
# ==============================
WORKDIR /app

# Install system dependencies (IMPORTANT for Wav2Lip)
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
# RUN HANDLER
# ==============================
CMD ["python", "-u", "handler.py"]