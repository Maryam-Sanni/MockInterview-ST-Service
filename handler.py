import runpod
import subprocess
import uuid
import os
import sys
import requests

import cloudinary
import cloudinary.uploader

# ==============================
# CONFIG
# ==============================

INPUT_DIR = "inputs"
OUTPUT_DIR = "/tmp"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# CLOUDINARY CONFIG
# ==============================

cloudinary.config(
    cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
    api_key=os.environ["CLOUDINARY_API_KEY"],
    api_secret=os.environ["CLOUDINARY_API_SECRET"],
    secure=True
)

def upload_to_cloudinary(file_path):
    result = cloudinary.uploader.upload_large(
        file_path,
        resource_type="video",
        folder="wav2lip"
    )
    return result["secure_url"]

# ==============================
# VOICES
# ==============================

def get_voice(language: str) -> str:
    voices = {
        "en": "en-GB-LibbyNeural",
        "nl": "nl-NL-ColetteNeural",
    }
    return voices.get(language, "en-GB-LibbyNeural")

# ==============================
# BASE VIDEO
# ==============================

def get_base_video():
    url = os.environ.get("BASE_VIDEO_URL")
    if not url:
        raise FileNotFoundError("BASE_VIDEO_URL not set")

    path = "/tmp/base.mp4"

    if not os.path.exists(path):
        r = requests.get(url)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)

    return path

# ==============================
# HANDLER
# ==============================

def handler(event):
    data = event["input"]

    text = data["text"]
    language = data.get("language", "en")

    job_id = str(uuid.uuid4())

    audio_path = os.path.join(OUTPUT_DIR, f"{job_id}.wav")
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")

    voice = get_voice(language)

    try:
        # 1. base video
        base_video = get_base_video()

        # 2. audio generation
        subprocess.run([
            sys.executable,
            "wav2lip/generate_audio.py",
            text,
            voice,
            audio_path
        ], check=True)

        # 3. Wav2Lip inference
        subprocess.run([
            sys.executable,
            "wav2lip/inference.py",
            "--checkpoint_path", "wav2lip/checkpoints/wav2lip_gan.pth",
            "--face", base_video,
            "--audio", audio_path,
            "--outfile", output_path,
            "--resize_factor", "2",
            "--nosmooth"
        ], check=True)

        # 4. upload to Cloudinary
        video_url = upload_to_cloudinary(output_path)

        return {
            "status": "success",
            "job_id": job_id,
            "video_url": video_url
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ==============================
# RUNPOD START
# ==============================

runpod.serverless.start({"handler": handler})