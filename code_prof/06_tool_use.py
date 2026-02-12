from dotenv import load_dotenv
from groq import Groq
from langfuse import observe, get_client
import json

load_dotenv()

groq_client = Groq()


# =============================================================================
# DEFINING TOOLS (as JSON schemas for the LLM)
# =============================================================================

# Tools are functions the LLM can decide to call.
# We describe them with a JSON schema so the LLM knows what's available.

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'Paris'"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression, e.g. '2 + 2 * 3'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    # ⚠️ FRAGILE TOOL — intentionally bad description to showcase LLM variability
    # The description says "a date" but doesn't specify the expected format.
    # The LLM might pass: "March 15, 2025", "2025-03-15", "15/03/2025", "tomorrow"…
    {
        "type": "function",
        "function": {
            "name": "get_bookings",
            "description": "Get restaurant bookings for a date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "A date"
                    }
                },
                "required": ["date"]
            }
        }
    },
]


# =============================================================================
# TOOL IMPLEMENTATIONS (the actual Python functions)
# =============================================================================

@observe()
def get_weather(city: str) -> str:
    """Simulate a weather API call."""
    # In a real app, this would call a weather API
    fake_data = {
        "Paris": "15°C, cloudy",
        "London": "12°C, rainy",
        "Tokyo": "22°C, sunny",
        "New York": "18°C, partly cloudy",
    }
    return fake_data.get(city, f"No weather data available for {city}")


@observe()
def calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        # Only allow safe math operations
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: only basic math operations are allowed"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@observe()
def get_bookings(date: str) -> str:
    """
    ⚠️ FRAGILE: only works with exact "DD/MM/YYYY" format.
    But the tool description just says "a date" — so the LLM can pass anything.
    Run Example 4 multiple times: you'll see the LLM use different formats!
    """
    import re
    if not re.match(r"^\d{2}/\d{2}/\d{4}$", date):
        return f"ERROR: invalid date format '{date}'. Expected DD/MM/YYYY."

    fake_bookings = {
        "15/03/2025": "2 bookings: Table for 4 at 19:00, Table for 2 at 20:30",
        "16/03/2025": "1 booking: Table for 6 at 20:00",
    }
    return fake_bookings.get(date, f"No bookings found for {date}")


# Registry mapping tool names to their implementations
TOOL_REGISTRY = {
    "get_weather": get_weather,
    "calculate": calculate,
    "get_bookings": get_bookings,
}


# =============================================================================
# THE TOOL-CALLING LOOP
# =============================================================================

@observe()
def tool_calling_agent(user_message: str) -> str:
    """
    A simple tool-calling loop:
    1. Send user message + tool definitions to the LLM
    2. If the LLM wants to call tools, execute them
    3. Send tool results back to the LLM
    4. Repeat until the LLM gives a final text response
    """

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the provided tools when needed to answer questions accurately."
        },
        {"role": "user", "content": user_message}
    ]

    for iteration in range(5):  # Max 5 iterations to avoid infinite loops
        print(f"\n  [Iteration {iteration + 1}]")

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=False,
        )

        message = response.choices[0].message

        # If no tool calls, the LLM is giving its final answer
        if not message.tool_calls:
            print(f"  Final answer ready.")
            return message.content

        # Process each tool call
        messages.append(message)  # Add assistant's tool-call message to history

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            print(f"  Tool call: {name}({args})")

            # Execute the tool
            func = TOOL_REGISTRY.get(name)
            if func:
                result = func(**args)
            else:
                result = f"Error: unknown tool '{name}'"

            print(f"  Result: {result}")

            # Add tool result to message history
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return "Error: max iterations reached"


# =============================================================================
# RUN EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TOOL USE / FUNCTION CALLING DEMO")
    print("=" * 60)

    # Example 1: Simple tool call
    print("\n--- Example 1: Weather query ---")
    answer = tool_calling_agent("What's the weather like in Paris?")
    print(f"\nAnswer: {answer}")

    # Example 2: Multiple tool calls
    print("\n--- Example 2: Multiple tools ---")
    answer = tool_calling_agent(
        "What's the weather in Tokyo and London? Also, what is 42 * 17 + 3?"
    )
    print(f"\nAnswer: {answer}")

    # Example 3: No tools needed
    print("\n--- Example 3: No tools needed ---")
    answer = tool_calling_agent("What is the capital of France?")
    print(f"\nAnswer: {answer}")

    # Example 4: ⚠️ FRAGILE TOOL — run this multiple times!
    # The LLM has to guess the date format because the tool description is vague.
    # You'll see it pass "2025-03-15", "March 15, 2025", "15/03/2025", etc.
    # Only "DD/MM/YYYY" works — everything else returns an error.
    #
    # LESSON: vague tool descriptions + strict implementations = unreliable agents.
    # Fix: make the description explicit (see get_weather for a good example).
    print("\n--- Example 4: Fragile tool (run multiple times!) ---")
    answer = tool_calling_agent("Show me the restaurant bookings for March 15th, 2025")
    print(f"\nAnswer: {answer}")

    get_client().flush()
