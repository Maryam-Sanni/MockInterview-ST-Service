from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import subprocess
import uuid
import os
import sys

# ==============================
# CONFIG
# ==============================

INPUT_DIR = "inputs"
OUTPUT_DIR = "results"
DEFAULT_BASE_VIDEO = os.environ.get(
    "WAV2LIP_BASE_VIDEO",
    os.path.join(INPUT_DIR, "video.mp4"),
)

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# APP INIT
# ==============================

app = FastAPI()

app.mount("/results", StaticFiles(directory=OUTPUT_DIR), name="results")

# ==============================
# REQUEST MODEL
# ==============================

class GenerateRequest(BaseModel):
    text: str
    language: str = "en"

# ==============================
# HELPERS
# ==============================

def get_voice(language: str) -> str:
    voices = {
        "en": "en-GB-LibbyNeural",
        "nl": "nl-NL-ColetteNeural",
    }
    return voices.get(language, "en-GB-LibbyNeural")

def get_base_video() -> str:
    candidates = [
        DEFAULT_BASE_VIDEO,
        os.path.join(INPUT_DIR, "vid.mp4"),
        os.path.join(INPUT_DIR, "vid3.mp4"),
        os.path.join(INPUT_DIR, "video3.mp4"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "No base video found. Set WAV2LIP_BASE_VIDEO or add inputs/video.mp4."
    )

# ==============================
# MAIN ENDPOINT
# ==============================

@app.post("/generate")
async def generate_video(req: GenerateRequest):
    job_id = str(uuid.uuid4())

    audio_path = os.path.join(INPUT_DIR, f"{job_id}.wav")
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")

    voice = get_voice(req.language)

    print(f"\n🎬 New Job: {job_id}")
    print(f"Text: {req.text}")
    print(f"Voice: {voice}")

    try:
        base_video = get_base_video()

        # ==============================
        # 1. Generate Audio
        # ==============================
        subprocess.run([
            sys.executable, "generate_audio.py",
            req.text,
            voice,
            audio_path
        ], check=True)

        print("✅ Audio generated")

        # ==============================
        # 2. Run Wav2Lip
        # ==============================
        subprocess.run([
    sys.executable, "inference.py",
    "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
    "--face", base_video,
    "--audio", audio_path,
    "--outfile", output_path,
    "--resize_factor", "2",
    "--nosmooth",
    "--pads", "0", "10", "0", "0",
    "--face_det_batch_size", "8",
    "--wav2lip_batch_size", "128"
        ], check=True)

        print("✅ Video generated")

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("❌ Error during processing:", e)
        return {"error": "Video generation failed"}

    # ==============================
    # RETURN VIDEO URL
    # ==============================
    return {
        "job_id": job_id,
        "video_url": f"http://localhost:8000/results/{job_id}.mp4"
    }

# ==============================
# HEALTH CHECK
# ==============================

@app.get("/")
def root():
    return {"status": "AI Video Server Running 🚀"}

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
