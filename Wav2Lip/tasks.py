import os
import uuid
from celery_app import celery
from generate_audio import generate_audio
from inference import generate_lip_sync

INPUT_DIR = "inputs"
OUTPUT_DIR = "results"
BASE_VIDEO = os.path.join(INPUT_DIR, "video.mp4")


@celery.task(bind=True)
def generate_video_task(self, text, language):
    job_id = str(uuid.uuid4())

    audio_path = os.path.join(INPUT_DIR, f"{job_id}.wav")
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")

    try:
        voice_map = {
            "en": "en-GB-LibbyNeural",
            "nl": "nl-NL-ColetteNeural",
        }
        voice = voice_map.get(language, "en-GB-LibbyNeural")

        # 1. TTS (NO subprocess)
        generate_audio(text, voice, audio_path)

        # 2. Lip sync (NO subprocess)
        generate_lip_sync(
            face_video=BASE_VIDEO,
            audio_path=audio_path,
            output_path=output_path
        )

        return {
            "job_id": job_id,
            "status": "completed",
            "video_url": f"/results/{job_id}.mp4"
        }

    except Exception as e:
        return {
            "job_id": job_id,
            "status": "failed",
            "error": str(e)
        }