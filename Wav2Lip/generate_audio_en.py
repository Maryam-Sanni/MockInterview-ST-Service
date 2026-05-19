import asyncio
import edge_tts

async def main():
    text = "Good day and welcome to this interview session. I will be guiding you through a series of questions to assess your experience and suitability for the role. Let us begin."

    communicate = edge_tts.Communicate(
        text,
        voice="en-GB-LibbyNeural",
        rate="+10%"
    )

    await communicate.save("inputs/audio.wav")
    print("Audio saved successfully")

if __name__ == "__main__":
    asyncio.run(main())