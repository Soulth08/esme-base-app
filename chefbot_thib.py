from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, tool
from langfuse import observe, get_client, Evaluation
from groq import Groq
import litellm
import json
from datetime import datetime

load_dotenv()

# --- Langfuse tracing for LiteLLM (v3 — OpenTelemetry) ---
litellm.callbacks = ["langfuse_otel"]

groq_client = Groq()
model = LiteLLMModel(model_id="groq/meta-llama/llama-4-scout-17b-16e-instruct")
def build_multi_agent_system():
    """
    Create a manager agent that delegates to specialized sub-agents.
    Each sub-agent has its own tools and expertise.
    """

    # --- Worker 1: Research Agent ---
    # Has web search capabilities
    ask_chef = CodeAgent(
        tools=[WebSearchTool(), VisitWebpageTool()],
        model=model,
        name="research_agent",
        description=(
            "tu est un chef cuisinier francais specialise en cuisine de saison"
        ),
        température=0.2,
    )

    manager = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[research_agent, analysis_agent],
        max_steps=8,
    )

    return manager