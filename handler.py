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
APP_DIR = os.path.dirname(os.path.abspath(__file__))
WAV2LIP_DIR = os.path.join(APP_DIR, "Wav2Lip")

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

def run_checked(command, *, cwd=None, timeout=None):
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        timeout=timeout,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        command_text = " ".join(command)
        stderr = result.stderr.strip()
        detail = f"{command_text} failed with exit code {result.returncode}"
        if stderr:
            detail = f"{detail}: {stderr}"
        raise RuntimeError(detail)

    return result

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
# BASE VIDEO DOWNLOAD (ROBUST)
# ==============================

def get_base_video():
    url = os.environ.get("BASE_VIDEO_URL")
    if not url:
        raise Exception("BASE_VIDEO_URL is not set")

    path = "/tmp/base.mp4"

    print("Downloading base video:", url)

    r = requests.get(url, timeout=60, stream=True, allow_redirects=True)
    r.raise_for_status()

    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    size = os.path.getsize(path)
    print("Base video size:", size)

    if size < 1000:
        raise Exception("Base video download failed (too small)")

    # validate video integrity
    ffprobe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", path],
        capture_output=True,
        text=True
    )

    if ffprobe.returncode != 0:
        raise Exception("Invalid base video (ffprobe failed)")

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
        print("🚀 JOB START:", job_id)
        print("TEXT:", text)
        print("VOICE:", voice)

        # =========================
        # 1. Base video
        # =========================
        base_video = get_base_video()

        # =========================
        # 2. Generate audio
        # =========================
        run_checked([
            sys.executable,
            "generate_audio.py",
            text,
            voice,
            audio_path
        ], cwd=WAV2LIP_DIR, timeout=300)

        # =========================
        # 3. Wav2Lip inference
        # =========================
        run_checked([
            sys.executable,
            "inference.py",
            "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
            "--face", base_video,
            "--audio", audio_path,
            "--outfile", output_path,
            "--resize_factor", "2",
            "--nosmooth"
        ], cwd=WAV2LIP_DIR, timeout=600)

        # =========================
        # 4. Upload to Cloudinary
        # =========================
        video_url = upload_to_cloudinary(output_path)

        return {
            "status": "success",
            "job_id": job_id,
            "video_url": video_url
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {
            "status": "error",
            "message": str(e)
        }

# ==============================
# RUNPOD START
# ==============================

runpod.serverless.start({"handler": handler})
