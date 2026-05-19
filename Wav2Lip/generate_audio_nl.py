import asyncio
import edge_tts

async def main():
    text = "Welkom bij je mock interview sessie met WinResponse. Ik begeleid je door een aantal vragen. Spreek duidelijk en neem je tijd. Laten we beginnen."

    communicate = edge_tts.Communicate(
        text,
        voice="nl-NL-FennaNeural",
        rate="+10%"
    )

    await communicate.save("inputs/audio.wav")
    print("Audio saved successfully")

if __name__ == "__main__":
    asyncio.run(main())