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

OUTPUT_DIR = "/tmp"
WAV2LIP_DIR = "/app/Wav2Lip"

CHECKPOINT = "/app/Wav2Lip/checkpoints/wav2lip_gan.pth"
S3FD = "/app/Wav2Lip/face_detection/detection/sfd/s3fd.pth"

INFERENCE_SCRIPT = os.path.join(WAV2LIP_DIR, "inference.py")
GENERATE_AUDIO_SCRIPT = os.path.join(WAV2LIP_DIR, "generate_audio.py")

BASE_VIDEO_CACHE = "/tmp/base.mp4"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🚀 RunPod Wav2Lip Serverless Ready")

# ==============================
# CLOUDINARY
# ==============================

cloudinary.config(
    cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
    api_key=os.environ["CLOUDINARY_API_KEY"],
    api_secret=os.environ["CLOUDINARY_API_SECRET"],
    secure=True
)

def upload_to_cloudinary(path):
    result = cloudinary.uploader.upload_large(
        path,
        resource_type="video",
        folder="wav2lip"
    )
    return result["secure_url"]

# ==============================
# HELPERS
# ==============================

def run_cmd(cmd, cwd=None, timeout=600):
    print("▶", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result


def validate(path):
    if not os.path.exists(path):
        raise Exception(f"Missing file: {path}")
    if os.path.getsize(path) < 1000:
        raise Exception(f"File too small: {path}")

# ==============================
# BASE VIDEO (CACHED ONLY ONCE PER CONTAINER)
# ==============================

def get_base_video():
    if os.path.exists(BASE_VIDEO_CACHE):
        return BASE_VIDEO_CACHE

    url = os.environ.get("BASE_VIDEO_URL")
    if not url:
        raise Exception("BASE_VIDEO_URL missing")

    print("Downloading base video once...")

    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(BASE_VIDEO_CACHE, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            f.write(chunk)

    return BASE_VIDEO_CACHE

# ==============================
# VOICE MAP
# ==============================

def get_voice(lang):
    return {
        "en": "en-GB-LibbyNeural",
        "nl": "nl-NL-ColetteNeural",
    }.get(lang, "en-GB-LibbyNeural")

# ==============================
# HANDLER
# ==============================

def handler(event):
    data = event.get("input", {})

    text = data["text"]
    language = data.get("language", "en")

    job_id = str(uuid.uuid4())

    audio_path = f"/tmp/{job_id}.wav"
    output_path = f"/tmp/{job_id}.mp4"

    voice = get_voice(language)

    try:
        print("================================")
        print("JOB:", job_id)
        print("TEXT:", text)
        print("VOICE:", voice)

        # =========================
        # 1. BASE VIDEO (CACHED)
        # =========================
        base_video = get_base_video()

        # =========================
        # 2. AUDIO GENERATION
        # =========================
        run_cmd([
            sys.executable,
            GENERATE_AUDIO_SCRIPT,
            text,
            voice,
            audio_path
        ], timeout=300)

        validate(audio_path)

        # =========================
        # 3. WAV2LIP INFERENCE
        # =========================
        run_cmd([
            sys.executable,
            INFERENCE_SCRIPT,
            "--checkpoint_path", CHECKPOINT,
            "--face", base_video,
            "--audio", audio_path,
            "--outfile", output_path,
            "--resize_factor", "2",
            "--nosmooth",
            "--face_det_batch_size", "1",
            "--wav2lip_batch_size", "16",
        ], cwd=WAV2LIP_DIR, timeout=600)

        validate(output_path)

        # =========================
        # 4. UPLOAD
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
# RUNPOD ENTRY
# ==============================

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler
    })