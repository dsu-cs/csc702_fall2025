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
DURATION = 5
CHANNELS = 1

# Select GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup speech recognition model
model = whisper.load_model("turbo", device=device)

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
def record_and_transcribe():
    print("Recording for 5 seconds...")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
    )
    sd.wait()
    print("Done recording.")

    np_audio = audio.squeeze()
    result = model.transcribe(np_audio, fp16=False)
    return result["text"].strip()


# Agent setup with tool declaration
root_agent = Agent(
    name="weather_time_agent",
    model="gemini-2.5-flash",
    description=(
        "Agent to answer questions about the time and weather in a city."
    ),
    instruction=(
        "You are a helpful agent who can answer user questions about the time and weather in a city."
    ),
    tools=[get_weather, get_current_time],
)

session_service = InMemorySessionService()

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
    runner = await init_runner()

    print("Speak into the microphone (say 'quit' to exit):")

    while True:
        # Run blocking audio + whisper in a worker thread
        user_input = await asyncio.to_thread(record_and_transcribe)

        print(f"User > {user_input}")

        if user_input.lower() == "quit": # User input should be a string if the user records audio. 
            break

        reply = await call_agent(runner, user_input)
        print("Agent >", reply)



if __name__ == "__main__":
    asyncio.run(main())