import asyncio
import os
import subprocess
import tempfile
import edge_tts


def generate_audio(text: str, voice: str, output_path: str):
    """
    Converts text to speech and saves it as an audio file
    """

    async def _generate():
        mp3_output = output_path
        if output_path.lower().endswith(".wav"):
            fd, mp3_output = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)

        communicate = edge_tts.Communicate(
            text=text,
            voice=voice
        )
        await communicate.save(mp3_output)

        if mp3_output != output_path:
            try:
                subprocess.run([
                    "ffmpeg",
                    "-y",
                    "-i", mp3_output,
                    "-ar", "16000",
                    "-ac", "1",
                    output_path,
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            finally:
                if os.path.exists(mp3_output):
                    os.remove(mp3_output)

    # run async TTS
    asyncio.run(_generate())

    return output_path


# Optional: allows you to test it manually
if __name__ == "__main__":
    import sys

    text = sys.argv[1]
    voice = sys.argv[2]
    output = sys.argv[3]

    generate_audio(text, voice, output)
