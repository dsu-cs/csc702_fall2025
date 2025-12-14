import asyncio
import datetime
from zoneinfo import ZoneInfo
from google.genai import types
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, DatabaseSessionService

USER_ID = "Joseph"
SESSION_ID = "1"

# Mock functions to use as tools
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

#session_service = InMemorySessionService()
session_service = DatabaseSessionService(db_url='sqlite+aiosqlite:///./agent_sessions.db')

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

    print("Enter messages (type 'quit' to exit):")

    while True:
        user_input = input("User > ")
        if user_input.lower() == "quit":
            break
        
        reply = await call_agent(runner, user_input)
        print("Agent > ", reply)


if __name__ == "__main__":
    asyncio.run(main())