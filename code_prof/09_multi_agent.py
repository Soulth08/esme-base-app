"""
Multi-Agent Systems
===================
Prerequisites:
    pip install 'smolagents[litellm,toolkit]' langfuse python-dotenv
    pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp

Hierarchical multi-agent: a manager delegates sub-tasks
to specialized worker agents, each with their own tools and memory.

Pattern: Manager → [Research Agent, Analysis Agent]
"""

from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, tool, WebSearchTool, VisitWebpageTool
from langfuse import observe, get_client
import litellm

load_dotenv()

# --- Langfuse tracing (v3 — OpenTelemetry) ---
# "langfuse_otel" creates OTel spans for each LLM call.
# Combined with @observe() on the run functions, LLM calls nest under
# the parent span instead of appearing as standalone traces.
litellm.callbacks = ["langfuse_otel"]

# Option A: Groq (free, fast, but rate-limited and weaker on multi-agent tasks)
#model = LiteLLMModel(model_id="groq/llama-3.3-70b-versatile")

# Option B: Gemini (requires GEMINI_API_KEY env variable)
model = LiteLLMModel(model_id="gemini/gemini-3-pro-preview")


# =============================================================================
# TOOLS FOR EACH SPECIALIZED AGENT
# =============================================================================

@tool
def analyze_sentiment(text: str) -> str:
    """
    Analyze the sentiment of a text. Returns positive, negative, or neutral
    with a short explanation.

    Args:
        text: The text to analyze.
    """
    # Simple keyword-based sentiment (in a real app, use an LLM or model)
    positive_words = {"good", "great", "excellent", "amazing", "love", "best", "happy", "wonderful"}
    negative_words = {"bad", "terrible", "worst", "hate", "awful", "poor", "disappointing"}

    words = set(text.lower().split())
    pos = len(words & positive_words)
    neg = len(words & negative_words)

    if pos > neg:
        return f"POSITIVE (matched {pos} positive keywords)"
    elif neg > pos:
        return f"NEGATIVE (matched {neg} negative keywords)"
    return "NEUTRAL (no strong sentiment detected)"


@tool
def summarize_points(points: str) -> str:
    """
    Take a list of points (one per line) and return a numbered summary.

    Args:
        points: Multiple points separated by newlines.
    """
    lines = [line.strip() for line in points.strip().split("\n") if line.strip()]
    summary = "\n".join(f"{i+1}. {line}" for i, line in enumerate(lines))
    return f"Summary ({len(lines)} points):\n{summary}"


# =============================================================================
# BUILDING THE MULTI-AGENT SYSTEM
# =============================================================================

def build_multi_agent_system():
    """
    Create a manager agent that delegates to specialized sub-agents.
    Each sub-agent has its own tools and expertise.
    """

    # --- Worker 1: Research Agent ---
    # Has web search capabilities
    research_agent = CodeAgent(
        tools=[WebSearchTool(), VisitWebpageTool()],
        model=model,
        name="research_agent",
        description=(
            "A research agent that can search the web for information. "
            "Give it a research question and it will find relevant data."
        ),
        max_steps=5,
    )

    # --- Worker 2: Analysis Agent ---
    # Has sentiment analysis and summarization tools
    analysis_agent = CodeAgent(
        tools=[analyze_sentiment, summarize_points],
        model=model,
        name="analysis_agent",
        description=(
            "An analysis agent that can analyze sentiment of text "
            "and summarize lists of points. Give it text to analyze."
        ),
        max_steps=5,
    )


    manager = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[research_agent, analysis_agent],
        max_steps=8,
    )

    return manager


# =============================================================================
# SIMPLE MULTI-AGENT (without web search)
# =============================================================================

def build_simple_multi_agent():
    """
    A simpler version that doesn't require web search.
    Good for testing without external dependencies.
    """

    @tool
    def get_product_reviews(product: str) -> str:
        """
        Get customer reviews for a product.

        Args:
            product: The product name to get reviews for.
        """
        fake_reviews = {
            "laptop": (
                "Great performance for the price\n"
                "Battery life is disappointing\n"
                "Excellent build quality\n"
                "The screen is amazing\n"
                "Keyboard could be better"
            ),
            "headphones": (
                "Best sound quality I've ever heard\n"
                "Comfortable for long sessions\n"
                "Noise cancellation is wonderful\n"
                "A bit expensive but worth it\n"
                "Love the wireless range"
            ),
        }
        return fake_reviews.get(
            product.lower(),
            f"No reviews found for '{product}'"
        )

    # Worker: fetches reviews
    review_agent = CodeAgent(
        tools=[get_product_reviews],
        model=model,
        name="review_fetcher",
        description="Fetches product reviews. Give it a product name. Returns a plain text string.",
        max_steps=3,
    )

    # Worker: analyzes text
    analysis_agent = CodeAgent(
        tools=[analyze_sentiment, summarize_points],
        model=model,
        name="text_analyzer",
        description="Analyzes sentiment and summarizes text. Give it text to analyze. Returns a plain text string.",
        max_steps=3,
    )

    # Manager: coordinates both
    manager = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[review_agent, analysis_agent],
        max_steps=8,
    )

    return manager


# =============================================================================
# RUN — @observe() wraps agent.run() so all LLM calls nest under one trace
# =============================================================================

@observe()
def run_simple_multi_agent():
    manager = build_simple_multi_agent()
    return manager.run(
        "Get the reviews for 'laptop', analyze the overall sentiment, "
        "and give me a summary of the key points."
    )


@observe()
def run_full_multi_agent():
    manager = build_multi_agent_system()
    return manager.run(
        "Research the pros and cons of electric vehicles in 2025, "
        "then analyze the overall sentiment and summarize the key points."
    )

if __name__ == "__main__":
    print("=" * 60)
    print("MULTI-AGENT SYSTEMS")
    print("=" * 60)

    # Simple version (no web search needed)
    print("\n--- Simple Multi-Agent: Product Review Analysis ---")
    result = run_simple_multi_agent()
    print(f"\nFinal result: {result}")

    # Full version with web search
    # print("\n--- Full Multi-Agent: Web Research + Analysis ---")
    # result = run_full_multi_agent()
    # print(f"\nFinal result: {result}")

    get_client().flush()
