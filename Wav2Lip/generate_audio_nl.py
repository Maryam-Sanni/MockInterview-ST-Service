import asyncio
import edge_tts

async def main():
    text = "We zullen doorgaan naar de volgende vraag."

    communicate = edge_tts.Communicate(
        text,
        voice="nl-NL-FennaNeural",
        rate="+10%"
    )

    await communicate.save("inputs/audio.wav")
    print("Audio saved successfully")

if __name__ == "__main__":
    asyncio.run(main())