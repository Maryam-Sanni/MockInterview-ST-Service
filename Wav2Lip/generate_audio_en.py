import asyncio
import edge_tts

async def main():
    text = "Okay, we’ll move forward to the next question."

    communicate = edge_tts.Communicate(
        text,
        voice="en-GB-LibbyNeural",
        rate="+10%"
    )

    await communicate.save("inputs/audio.wav")
    print("Audio saved successfully")

if __name__ == "__main__":
    asyncio.run(main())