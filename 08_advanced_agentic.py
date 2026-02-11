"""
Advanced Agentic Patterns
=========================
Prerequisites:
    pip install 'smolagents[litellm,toolkit]' langfuse python-dotenv
    pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp

Building on 07, this file covers:
- Custom Tool class (vs @tool decorator)
- Agent with planning (periodic re-planning between steps)
- Agent with custom instructions
- Conversational agent (memory across turns)
"""

from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, tool, Tool, WebSearchTool
from langfuse import observe, get_client
import litellm

load_dotenv()

# --- Langfuse tracing for LiteLLM (v3 — OpenTelemetry) ---
litellm.callbacks = ["langfuse_otel"]

model = LiteLLMModel(model_id="groq/meta-llama/llama-4-scout-17b-16e-instruct")


# =============================================================================
# CUSTOM TOOL CLASS (more control than @tool decorator)
# =============================================================================

class DatabaseLookupTool(Tool):
    """
    Use the Tool class when you need:
    - Initialization logic (DB connections, API clients)
    - Multiple helper methods
    - Class-level state
    """
    name = "database_lookup"
    description = "Look up a product by name in the database. Returns price and stock."
    inputs = {
        "product_name": {
            "type": "string",
            "description": "The name of the product to look up."
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        # Simulate a database
        self.products = {
            "laptop": {"price": 999.99, "stock": 15},
            "keyboard": {"price": 79.99, "stock": 142},
            "monitor": {"price": 349.99, "stock": 38},
            "mouse": {"price": 29.99, "stock": 200},
            "headphones": {"price": 149.99, "stock": 67},
        }

    def forward(self, product_name: str) -> str:
        product_name = product_name.lower().strip()
        product = self.products.get(product_name)
        if product:
            return f"{product_name}: ${product['price']}, {product['stock']} in stock"
        available = ", ".join(self.products.keys())
        return f"Product '{product_name}' not found. Available: {available}"


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a math expression and return the result.

    Args:
        expression: A math expression, e.g. '999.99 * 3 + 79.99'.
    """
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: only basic math operations allowed"
    return str(eval(expression))


# =============================================================================
# AGENT WITH PLANNING
# =============================================================================

@observe()
def run_planning_agent():
    """
    planning_interval=2 makes the agent pause every 2 steps to:
    - Reflect on what it knows so far
    - Update its plan for remaining steps
    This helps on complex, multi-step tasks.
    """

    agent = CodeAgent(
        tools=[DatabaseLookupTool(), calculate],
        model=model,
        planning_interval=2,  # Re-plan every 2 steps
        max_steps=8,
    )

    result = agent.run(
        "I want to buy 2 laptops and 3 keyboards. "
        "Look up each product, then calculate the total cost."
    )

    print(f"Result: {result}")
    return result


# =============================================================================
# AGENT WITH CUSTOM INSTRUCTIONS
# =============================================================================

@observe()
def run_instructed_agent():
    """
    Custom instructions are appended to the system prompt.
    Use them to set tone, constraints, or domain-specific rules.
    """

    agent = CodeAgent(
        tools=[DatabaseLookupTool(), calculate],
        model=model,
        instructions=(
            "You are a shopping assistant for an electronics store. "
            "Always greet the customer, show prices in USD, "
            "and suggest related products when relevant."
        ),
        max_steps=5,
    )

    result = agent.run("I need a new mouse, what do you have?")

    print(f"Result: {result}")
    return result


# =============================================================================
# CONVERSATIONAL AGENT (memory across turns)
# =============================================================================

@observe()
def run_conversational_agent():
    """
    Pass reset=False to keep the agent's memory between runs.
    This enables multi-turn conversations.
    """

    agent = CodeAgent(
        tools=[DatabaseLookupTool(), calculate],
        model=model,
        max_steps=5,
    )

    # Turn 1
    print("  User: What's the price of a monitor?")
    result1 = agent.run("What's the price of a monitor?")
    print(f"  Agent: {result1}")

    # Turn 2 — agent remembers the previous exchange
    print("\n  User: And how much for 3 of them?")
    result2 = agent.run(result1 + "And how much for 3 of them?", reset=False)
    print(f"  Agent: {result2}")

    return result2


# =============================================================================
# AGENT WITH WEB SEARCH
# =============================================================================

@observe()
def run_web_search_agent():
    """
    Using the built-in WebSearchTool (DuckDuckGo).
    Requires: pip install 'smolagents[toolkit]'
    """

    agent = CodeAgent(
        tools=[WebSearchTool()],
        model=model,
        max_steps=5,
    )

    result = agent.run("What is the current population of France?")

    print(f"Result: {result}")
    return result


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ADVANCED AGENTIC PATTERNS")
    print("=" * 60)

    print("\n--- 1. Agent with Planning ---")
    run_planning_agent()

    print("\n--- 2. Agent with Custom Instructions ---")
    run_instructed_agent()

    print("\n--- 3. Conversational Agent (multi-turn) ---")
    run_conversational_agent()

    # Uncomment to try web search (requires toolkit extra)
    # print("\n--- 4. Agent with Web Search ---")
    # run_web_search_agent()

    get_client().flush()
