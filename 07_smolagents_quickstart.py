"""
SmolAgents Quickstart
=====================
Prerequisites:
    pip install 'smolagents[litellm,telemetry]' langfuse python-dotenv
    pip install opentelemetry-sdk opentelemetry-exporter-otlp openinference-instrumentation-smolagents

This file shows how to go from raw tool-calling (06) to a proper agent
framework that handles the loop, errors, and retries for you.

We use LiteLLMModel to connect smolagents to Groq.
"""

from dotenv import load_dotenv
from smolagents import CodeAgent, ToolCallingAgent, LiteLLMModel, tool
from langfuse import get_client

load_dotenv()

# model = LiteLLMModel(model_id="groq/llama-3.3-70b-versatile")
model = LiteLLMModel(model_id="gemini/gemini-3-pro-preview")


@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a given city.
    Returns a short weather description.

    Args:
        city: The city name, e.g. 'Paris', 'Tokyo'.
    """
    
    fake_data = {
        "Paris": "15°C, cloudy",
        "London": "12°C, rainy",
        "Tokyo": "22°C, sunny",
        "New York": "18°C, partly cloudy",
    }
    return fake_data.get(city, f"No weather data available for {city}")


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a math expression and return the result as a string.

    Args:
        expression: A math expression using +, -, *, /, e.g. '42 * 17 + 3'.
    """
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: only basic math operations are allowed"
    return str(eval(expression))


def run_code_agent():
    agent = CodeAgent(
        tools=[get_weather, calculate],
        model=model,
        max_steps=5,
    )

    result = agent.run(
        "What's the weather in Paris and Tokyo? "
        "Also compute the average of 15 and 22."
    )

    print(f"Result: {result}")
    return result


def run_tool_calling_agent():
    """
    ToolCallingAgent uses structured JSON to call tools.
    More predictable: validated inputs, no code execution risk.
    """

    agent = ToolCallingAgent(
        tools=[get_weather, calculate],
        model=model,
        max_steps=5,
    )

    result = agent.run("What is 1234 * 5678?")

    print(f"Result: {result}")
    return result



run_code_agent()

# print("\n--- ToolCallingAgent (structured JSON calls) ---")
# run_tool_calling_agent()

get_client().flush()
