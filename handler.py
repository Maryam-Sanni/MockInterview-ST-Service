import os
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import cloudinary
import cloudinary.uploader
import requests
import runpod


ROOT_DIR = Path(__file__).resolve().parent
WAV2LIP_DIR = ROOT_DIR / "Wav2Lip"
CHECKPOINT_PATH = Path(
    os.environ.get("WAV2LIP_CHECKPOINT_PATH", WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth")
)
WORK_DIR = Path(os.environ.get("RUNPOD_WORK_DIR", "/tmp/wav2lip_jobs"))
BASE_VIDEO_CACHE = WORK_DIR / "base.mp4"

DEFAULT_TIMEOUTS = {
    "download": int(os.environ.get("DOWNLOAD_TIMEOUT_SECONDS", "120")),
    "audio": int(os.environ.get("AUDIO_TIMEOUT_SECONDS", "300")),
    "inference": int(os.environ.get("INFERENCE_TIMEOUT_SECONDS", "900")),
    "upload": int(os.environ.get("UPLOAD_TIMEOUT_SECONDS", "600")),
}

VOICE_MAP = {
    "en": "en-GB-LibbyNeural",
    "nl": "nl-NL-ColetteNeural",
}

WORK_DIR.mkdir(parents=True, exist_ok=True)
(WAV2LIP_DIR / "temp").mkdir(parents=True, exist_ok=True)

# Wav2Lip writes temp/result.avi, so keep inference single-file safe per worker.
INFERENCE_LOCK = threading.Lock()


class HandlerError(Exception):
    pass


def get_voice(language: str, requested_voice: str | None = None) -> str:
    if requested_voice:
        return requested_voice
    return VOICE_MAP.get((language or "en").lower(), VOICE_MAP["en"])


def run_command(command, *, cwd=None, timeout=None):
    print("Running:", " ".join(map(str, command)))
    completed = subprocess.run(
        [str(part) for part in command],
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    if completed.returncode != 0:
        raise HandlerError(f"Command failed with exit code {completed.returncode}: {command[0]}")
    return completed


def validate_video(path: Path):
    if not path.exists():
        raise HandlerError(f"Video file does not exist: {path}")
    if path.stat().st_size < 1000:
        raise HandlerError(f"Video file is too small or empty: {path}")

    run_command(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", path],
        timeout=30,
    )


def download_file(url: str, destination: Path, *, timeout: int):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".download")

    print(f"Downloading {url} to {destination}")
    with requests.get(url, timeout=timeout, stream=True, allow_redirects=True) as response:
        response.raise_for_status()
        with temp_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    temp_path.replace(destination)
    return destination


def get_base_video(data: dict, job_dir: Path) -> Path:
    input_video = data.get("base_video_path") or data.get("video_path")
    if input_video:
        path = Path(input_video)
        validate_video(path)
        return path

    request_url = data.get("base_video_url") or data.get("video_url")
    env_url = os.environ.get("BASE_VIDEO_URL")
    url = request_url or env_url
    if not url:
        raise HandlerError("Provide input.base_video_url or set BASE_VIDEO_URL")

    if request_url:
        request_video = job_dir / "base.mp4"
        download_file(request_url, request_video, timeout=DEFAULT_TIMEOUTS["download"])
        validate_video(request_video)
        return request_video

    if BASE_VIDEO_CACHE.exists() and BASE_VIDEO_CACHE.stat().st_size >= 1000:
        validate_video(BASE_VIDEO_CACHE)
        return BASE_VIDEO_CACHE

    download_file(url, BASE_VIDEO_CACHE, timeout=DEFAULT_TIMEOUTS["download"])
    validate_video(BASE_VIDEO_CACHE)
    return BASE_VIDEO_CACHE


def configure_cloudinary():
    required = ["CLOUDINARY_CLOUD_NAME", "CLOUDINARY_API_KEY", "CLOUDINARY_API_SECRET"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise HandlerError(f"Missing Cloudinary environment variables: {', '.join(missing)}")

    cloudinary.config(
        cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
        api_key=os.environ["CLOUDINARY_API_KEY"],
        api_secret=os.environ["CLOUDINARY_API_SECRET"],
        secure=True,
    )


def upload_to_cloudinary(file_path: Path, folder: str = "wav2lip") -> str:
    configure_cloudinary()
    result = cloudinary.uploader.upload_large(
        str(file_path),
        resource_type="video",
        folder=folder,
        timeout=DEFAULT_TIMEOUTS["upload"],
    )
    return result["secure_url"]


def build_inference_args(data: dict, base_video: Path, audio_path: Path, output_path: Path):
    args = [
        sys.executable,
        "inference.py",
        "--checkpoint_path",
        CHECKPOINT_PATH,
        "--face",
        base_video,
        "--audio",
        audio_path,
        "--outfile",
        output_path,
        "--resize_factor",
        str(data.get("resize_factor", os.environ.get("WAV2LIP_RESIZE_FACTOR", "2"))),
        "--crf",
        str(data.get("crf", os.environ.get("WAV2LIP_CRF", "16"))),
        "--preset",
        str(data.get("preset", os.environ.get("WAV2LIP_PRESET", "slow"))),
    ]

    if data.get("nosmooth", True):
        args.append("--nosmooth")
    if data.get("mouth_only", False):
        args.append("--mouth_only")
    if data.get("sharpen") is not None:
        args.extend(["--sharpen", str(data["sharpen"])])

    return args


def cleanup_job_dir(job_dir: Path):
    if os.environ.get("KEEP_RUNPOD_JOB_FILES", "").lower() in {"1", "true", "yes"}:
        return
    shutil.rmtree(job_dir, ignore_errors=True)


def handler(event):
    data = event.get("input") or {}
    job_id = data.get("job_id") or str(uuid.uuid4())
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    text = data.get("text")
    if not text:
        return {"status": "error", "job_id": job_id, "message": "input.text is required"}

    if not CHECKPOINT_PATH.exists():
        return {
            "status": "error",
            "job_id": job_id,
            "message": f"Wav2Lip checkpoint not found: {CHECKPOINT_PATH}",
        }

    language = data.get("language", "en")
    voice = get_voice(language, data.get("voice"))
    audio_path = job_dir / "speech.wav"
    output_path = job_dir / "result.mp4"

    try:
        print(f"JOB START: {job_id}")
        print(f"Language: {language}; voice: {voice}")

        base_video = get_base_video(data, job_dir)

        run_command(
            [sys.executable, WAV2LIP_DIR / "generate_audio.py", text, voice, audio_path],
            timeout=DEFAULT_TIMEOUTS["audio"],
        )

        with INFERENCE_LOCK:
            run_command(
                build_inference_args(data, base_video, audio_path, output_path),
                cwd=WAV2LIP_DIR,
                timeout=DEFAULT_TIMEOUTS["inference"],
            )

        if not output_path.exists() or output_path.stat().st_size < 1000:
            raise HandlerError("Wav2Lip did not produce a valid output video")

        upload = data.get("upload", True)
        result = {
            "status": "success",
            "job_id": job_id,
            "output_size": output_path.stat().st_size,
        }
        if upload:
            result["video_url"] = upload_to_cloudinary(
                output_path,
                folder=data.get("cloudinary_folder", "wav2lip"),
            )
        else:
            result["output_path"] = str(output_path)

        return result

    except Exception as exc:
        print(f"ERROR [{job_id}]: {exc}", file=sys.stderr)
        return {"status": "error", "job_id": job_id, "message": str(exc)}
    finally:
        if data.get("cleanup", True) and data.get("upload", True):
            cleanup_job_dir(job_dir)


runpod.serverless.start({"handler": handler})
