FROM runpod/pytorch:3.10-2.0.1-120-devel

# ==============================
# SYSTEM SETUP
# ==============================
WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ==============================
# PYTHON DEPENDENCIES (CACHE OPTIMIZED)
# ==============================
COPY requirements.txt /app/

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ==============================
# COPY PROJECT (LAST for caching)
# ==============================
COPY . /app

# ==============================
# RUN HANDLER
# ==============================
CMD ["python", "-u", "handler.py"]