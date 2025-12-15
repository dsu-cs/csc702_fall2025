import numpy
import torch
import whisper
import sounddevice as sd
import asyncio
import datetime
from zoneinfo import ZoneInfo
from google.genai import types
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# For in memory session management
USER_ID = "Joseph"
SESSION_ID = "1"

# Configurations for audio
SAMPLE_RATE = 16000
# DURATION = 5
CHANNELS = 1

CHUNK_MS = 30                 # audio chunk size
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000

SPEECH_THRESHOLD = 0.01       # adjust if needed
SILENCE_CHUNKS = 30           # ~450ms of silence
MAX_RECORD_SECONDS = 15


# Select GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup speech recognition model
model = whisper.load_model("turbo", device=device)

# User conversations, in memory for testing and development. 
session_service = InMemorySessionService()

# Mock functions to use as tools, these get registered with the agent in the initializer: tools=[get_weather, get_current_time]
def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": (
                "The weather in New York is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """

    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {
            "status": "error",
            "error_message": (
                f"Sorry, I don't have timezone information for {city}."
            ),
        }

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = (
        f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    )
    return {"status": "success", "report": report}

# Record audio and return the output from speech to text 
# def record_and_transcribe():
#     print("Recording for 5 seconds...")
#     audio = sd.rec(
#         int(DURATION * SAMPLE_RATE),
#         samplerate=SAMPLE_RATE,
#         channels=CHANNELS,
#         dtype="float32",
#     )
#     sd.wait()
#     print("Done recording.")

#     np_audio = audio.squeeze()
#     result = model.transcribe(np_audio, fp16=False)
#     return result["text"]

def measure_noise(seconds=2):
    print("Stay quiet...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    rms = numpy.sqrt(numpy.mean(audio.squeeze() ** 2))
    print(f"Noise RMS: {rms:.4f}")
    return rms


def record_until_silence():
    print("Listening...")

    audio_chunks = []
    speaking = False
    silent_chunks = 0
    total_chunks = 0

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=CHUNK_SAMPLES,
        dtype="float32",
    ) as stream:
        while True:
            chunk, _ = stream.read(CHUNK_SAMPLES)
            chunk = chunk.squeeze()

            rms = numpy.sqrt(numpy.mean(chunk ** 2))
            total_chunks += 1

            if rms > SPEECH_THRESHOLD:
                speaking = True
                silent_chunks = 0
                audio_chunks.append(chunk)
            else:
                if speaking:
                    silent_chunks += 1
                    audio_chunks.append(chunk)

            # Stop after speech followed by silence
            if speaking and silent_chunks >= SILENCE_CHUNKS:
                break

            # Safety: stop after max duration
            if total_chunks * CHUNK_MS >= MAX_RECORD_SECONDS * 1000:
                break

    print("Speech ended.")
    return numpy.concatenate(audio_chunks)



def transcribe_audio(audio):
    result = model.transcribe(audio, fp16=False)
    return result["text"]


# Agent setup with tool declaration
root_agent = Agent(
    name="weather_time_agent",
    model="gemini-2.5-flash-lite",
    description=(
        "Agent to answer questions about the time and weather in a city."
    ),
    instruction=(
        "You are a helpful agent who can answer user questions about the time and weather in a city."
    ),
    tools=[get_weather, get_current_time],
)

async def init_runner():
    
    await session_service.create_session(
        app_name = "voice_app",
        user_id = USER_ID,
        session_id = SESSION_ID
    )

    runner = Runner(
        agent = root_agent,
        app_name = "voice_app",
        session_service = session_service
    )

    return runner


async def call_agent(runner: Runner, user_text: str):
    content = types.Content(role = "user", parts = [types.Part(text = user_text)])

    final_text = None

    # Run the agent
    async for event in runner.run_async(
        user_id = USER_ID,
        session_id = SESSION_ID,
        new_message = content
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_text = event.content.parts[0].text
                break

    return final_text


async def main():
    global SPEECH_THRESHOLD

    runner = await init_runner()

    # Calibrate noise level
    noise = measure_noise()
    SPEECH_THRESHOLD = noise * 3

    print("Speak into the microphone (say 'quit' to exit):")

    # Terminate loop after 5 tries, this is just to deal with microphone bugginess. 
    count = 0

    while count < 5:

        # Run blocking audio + whisper in a worker thread
        audio = await asyncio.to_thread(record_until_silence)
        user_input = transcribe_audio(audio)

        print(f"User > {user_input}")

        # May need to keyboard interupt this exit condition. 
        # The whisper model seems to put periods at the end of single word sentences, during testing this often failed without the period because it's not an exact match. 
        if user_input.strip().lower() == "quit." or user_input.strip().lower() == 'quits.': # User input should be a string if the user records audio. 
            break

        reply = await call_agent(runner, user_input)
        print("Agent >", reply)

        count +=1



if __name__ == "__main__":
    asyncio.run(main())