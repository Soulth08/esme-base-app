"""
Evaluating Agentic Systems
===========================
Prerequisites:
    pip install 'smolagents[litellm]' langfuse python-dotenv
    pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp

Ties everything together: run an agent on a Langfuse dataset,
evaluate its outputs with an LLM judge, and compare configurations.

Builds on:
- 04 (datasets & experiments)
- 05 (LLM as a judge)
- 07-08 (smolagents)
"""

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


# =============================================================================
# TOOLS FOR THE AGENT UNDER TEST
# =============================================================================

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search an internal knowledge base for information.

    Args:
        query: The search query.
    """
    # Simulated knowledge base
    kb = {
        "return policy": "Items can be returned within 30 days with receipt. Electronics have a 15-day window.",
        "shipping": "Free shipping on orders over $50. Standard delivery takes 3-5 business days.",
        "warranty": "All electronics come with a 1-year manufacturer warranty. Extended warranty available for $29.99.",
        "payment": "We accept credit cards, PayPal, and Apple Pay. Installment plans available for orders over $200.",
        "hours": "Customer service is available Mon-Fri 9am-6pm EST. Chat support 24/7.",
    }
    for key, value in kb.items():
        if key in query.lower():
            return value
    return "No relevant information found. Try rephrasing your query."


@tool
def check_order_status(order_id: str) -> str:
    """
    Check the status of a customer order.

    Args:
        order_id: The order ID to look up, e.g. 'ORD-1234'.
    """
    fake_orders = {
        "ORD-1001": "Shipped - arriving in 2 days",
        "ORD-1002": "Processing - will ship tomorrow",
        "ORD-1003": "Delivered on Jan 15, 2025",
    }
    return fake_orders.get(order_id, f"Order '{order_id}' not found")


# =============================================================================
# CREATE THE EVALUATION DATASET
# =============================================================================

def create_agent_dataset():
    """Create a dataset of customer support questions with expected behaviors."""

    dataset = get_client().create_dataset(
        name="agent-eval-v1",
        description="Customer support agent evaluation dataset",
        metadata={"domain": "e-commerce_support"}
    )

    test_cases = [
        {
            "input": {"question": "What is your return policy for electronics?"},
            "expected_output": {
                "should_use_tool": True,
                "tool_name": "search_knowledge_base",
                "must_mention": ["15-day", "electronics"],
            }
        },
        {
            "input": {"question": "Where is my order ORD-1001?"},
            "expected_output": {
                "should_use_tool": True,
                "tool_name": "check_order_status",
                "must_mention": ["shipped", "2 days"],
            }
        },
        {
            "input": {"question": "Do you offer free shipping?"},
            "expected_output": {
                "should_use_tool": True,
                "tool_name": "search_knowledge_base",
                "must_mention": ["$50", "free shipping"],
            }
        },
        {
            "input": {"question": "Hello, how are you?"},
            "expected_output": {
                "should_use_tool": False,
                "must_mention": [],
            }
        },
        {
            "input": {"question": "What warranty do you offer on electronics?"},
            "expected_output": {
                "should_use_tool": True,
                "tool_name": "search_knowledge_base",
                "must_mention": ["1-year", "warranty"],
            }
        },
    ]

    for case in test_cases:
        get_client().create_dataset_item(
            dataset_name="agent-eval-v1",
            input=case["input"],
            expected_output=case["expected_output"],
        )

    print(f"Created dataset with {len(test_cases)} test cases")
    return dataset


# =============================================================================
# THE AGENT UNDER TEST
# =============================================================================

def build_support_agent():
    """Build the customer support agent we want to evaluate."""

    return CodeAgent(
        tools=[search_knowledge_base, check_order_status],
        model=model,
        instructions=(
            "You are a helpful customer support agent for an e-commerce store. "
            "Use the available tools to find accurate information before answering. "
            "Be concise and friendly."
        ),
        max_steps=5,
    )


# =============================================================================
# LLM JUDGE FOR AGENT OUTPUTS
# =============================================================================

AGENT_JUDGE_PROMPT = """You are evaluating a customer support AI agent.

Given:
- The customer's question
- The agent's response
- What the response should mention

Score on each criterion from 0.0 to 1.0:

1. **completeness**: Does the response contain all required information?
   - 1.0 = all key points mentioned, 0.5 = some missing, 0.0 = none mentioned
2. **helpfulness**: Is the response helpful and actionable for the customer?
   - 1.0 = very helpful, 0.5 = somewhat helpful, 0.0 = not helpful
3. **tone**: Is the response professional and friendly?
   - 1.0 = excellent tone, 0.5 = acceptable, 0.0 = rude or robotic

Respond ONLY with JSON:
{
    "completeness": 0.0,
    "helpfulness": 0.0,
    "tone": 0.0,
    "explanation": "brief justification"
}"""


@observe(name="agent-judge", as_type="generation")
def judge_agent_response(question: str, response: str, expected: dict) -> dict:
    """Use an LLM to evaluate the agent's response."""

    must_mention = expected.get("must_mention", [])

    result = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": AGENT_JUDGE_PROMPT},
            {"role": "user", "content": (
                f"Customer question: {question}\n\n"
                f"Agent response: {response}\n\n"
                f"Should mention: {', '.join(must_mention) if must_mention else 'N/A (greeting)'}"
            )}
        ],
        temperature=0.1,
    )

    return json.loads(result.choices[0].message.content)


# =============================================================================
# RUN THE EVALUATION EXPERIMENT
# =============================================================================

def run_agent_evaluation():
    """Run the agent on each dataset item and evaluate with the LLM judge."""

    dataset = get_client().get_dataset("agent-eval-v1")
    agent = build_support_agent()

    def task(*, item) -> str:
        # Run the agent on the question — reset memory for each item
        return str(agent.run(item.input["question"]))

    def evaluator(**kwargs) -> list:
        output = kwargs.get("output", "")
        expected = kwargs.get("expected_output", {})
        input_data = kwargs.get("input", {})

        scores = judge_agent_response(
            question=input_data.get("question", ""),
            response=output,
            expected=expected,
        )

        print(f"  Judge: {scores.get('explanation', '')[:80]}")

        return [
            Evaluation(name="completeness", value=scores["completeness"],
                       comment=scores.get("explanation")),
            Evaluation(name="helpfulness", value=scores["helpfulness"]),
            Evaluation(name="tone", value=scores["tone"]),
        ]

    results = get_client().run_experiment(
        name=f"agent-eval-{datetime.now().strftime('%H%M%S')}",
        data=dataset.items,
        task=task,
        evaluators=[evaluator],
        description="Customer support agent evaluation with LLM judge",
        metadata={
            "agent_model": "groq/llama-3.3-70b-versatile",
            "judge_model": "llama-3.3-70b-versatile",
        },
    )

    print("\nExperiment complete!")
    return results




# Step 1: Create dataset (only once)
print("\n--- Creating Dataset --- (uncomment for first run)")
#create_agent_dataset()

# Step 2: Run evaluation
print("\n--- Running Agent Evaluation ---")
results = run_agent_evaluation()

get_client().flush()
