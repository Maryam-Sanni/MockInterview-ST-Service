import runpod
import subprocess
import uuid
import os
import sys

INPUT_DIR = "inputs"
OUTPUT_DIR = "/tmp"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_voice(language: str) -> str:
    voices = {
        "en": "en-GB-LibbyNeural",
        "nl": "nl-NL-ColetteNeural",
    }
    return voices.get(language, "en-GB-LibbyNeural")


def get_base_video():
    candidates = [
        os.path.join(INPUT_DIR, "video.mp4"),
        os.path.join(INPUT_DIR, "vid.mp4"),
    ]

    for c in candidates:
        if os.path.isfile(c):
            return c

    raise FileNotFoundError("No base video found")


def handler(event):
    data = event["input"]

    text = data["text"]
    language = data.get("language", "en")

    job_id = str(uuid.uuid4())

    audio_path = os.path.join(OUTPUT_DIR, f"{job_id}.wav")
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")

    voice = get_voice(language)

    try:
        base_video = get_base_video()

        # =========================
        # 1. Generate Audio (FIXED PATH)
        # =========================
        subprocess.run([
            sys.executable,
            "wav2lip/generate_audio.py",
            text,
            voice,
            audio_path
        ], check=True)

        # =========================
        # 2. Run Wav2Lip (FIXED PATH)
        # =========================
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

        return {
            "status": "success",
            "job_id": job_id,
            "video_path": output_path
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


runpod.serverless.start({"handler": handler})