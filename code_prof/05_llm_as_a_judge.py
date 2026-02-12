from dotenv import load_dotenv
from langfuse import observe, get_client, Evaluation
from groq import Groq
import json
from datetime import datetime

load_dotenv()

groq_client = Groq()


# =============================================================================
# THE TASK TO EVALUATE (same as 04)
# =============================================================================

@observe()
def sentiment_task(text: str) -> dict:
    """Analyze sentiment of a text. This is the function we want to judge."""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": """Analyze sentiment and respond in JSON:
{
    "sentiment": "positive|negative|neutral|mixed",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}"""
            },
            {"role": "user", "content": text}
        ],
        temperature=0.2
    )

    return json.loads(response.choices[0].message.content)


# =============================================================================
# LLM AS A JUDGE
# =============================================================================

JUDGE_PROMPT = """You are an impartial judge evaluating a sentiment analysis system.

Given:
- The original text
- The system's output (sentiment, confidence, reasoning)
- The expected sentiment

Score the output on each criterion from 0.0 to 1.0:

1. **correctness**: Does the predicted sentiment match the expected one?
   - 1.0 = exact match, 0.5 = close (e.g. "mixed" when expected "positive"), 0.0 = wrong
2. **reasoning_quality**: Is the reasoning clear, specific, and well-justified?
   - 1.0 = excellent, 0.5 = acceptable, 0.0 = poor or missing
3. **confidence_calibration**: Is the confidence score appropriate for this text?
   - 1.0 = well-calibrated, 0.5 = slightly off, 0.0 = wildly wrong

Respond ONLY with JSON:
{
    "correctness": 0.0,
    "reasoning_quality": 0.0,
    "confidence_calibration": 0.0,
    "explanation": "brief justification for your scores"
}"""


@observe(name="llm-judge", as_type="generation")
def llm_judge(input_text: str, output: dict, expected_output: dict) -> dict:
    """Use an LLM to evaluate the quality of another LLM's output."""

    user_message = f"""Original text: "{input_text}"

System output:
- Sentiment: {output.get("sentiment")}
- Confidence: {output.get("confidence")}
- Reasoning: {output.get("reasoning", "N/A")}

Expected sentiment: {expected_output.get("sentiment")}"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.1  # Low temperature for consistent judging
    )

    return json.loads(response.choices[0].message.content)


# =============================================================================
# RUNNING THE EXPERIMENT WITH LLM JUDGE
# =============================================================================

def run_llm_judge_experiment():
    """
    Run an experiment using an LLM as evaluator instead of programmatic rules.
    Reuses the dataset created in 04_dataset_experiment.py.
    """

    dataset = get_client().get_dataset("sentiment-benchmark-v1")

    # Task: run sentiment analysis on each item
    def task(*, item) -> dict:
        return sentiment_task(item.input["text"])

    # Evaluator: use the LLM judge to score each output
    def llm_evaluator(**kwargs) -> list:
        output = kwargs.get("output")
        expected_output = kwargs.get("expected_output")
        input_data = kwargs.get("input")

        scores = llm_judge(
            input_text=input_data["text"],
            output=output,
            expected_output=expected_output
        )

        print(f"  Judge says: {scores.get('explanation', '')[:80]}")

        return [
            Evaluation(
                name="correctness",
                value=scores["correctness"],
                comment=scores.get("explanation")
            ),
            Evaluation(
                name="reasoning_quality",
                value=scores["reasoning_quality"],
            ),
            Evaluation(
                name="confidence_calibration",
                value=scores["confidence_calibration"],
            ),
        ]

    results = get_client().run_experiment(
        name=f"llm-judge-exp-{datetime.now().strftime('%H%M%S')}",
        data=dataset.items,
        task=task,
        evaluators=[llm_evaluator],
        description="Sentiment analysis evaluated by LLM judge",
        metadata={"model": "openai/gpt-oss-120b", "judge_model": "openai/gpt-oss-120b"},
    )

    print("\nExperiment complete! Check Langfuse UI for detailed judge scores.")
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("LLM AS A JUDGE DEMO")
    print("=" * 60)
    print("\nPrerequisite: run 04_dataset_experiment.py first to create the dataset.\n")

    print("--- Running Experiment with LLM Judge ---")
    try:
        results = run_llm_judge_experiment()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure dataset 'sentiment-benchmark-v1' exists (run 04 first).")

    get_client().flush()
    print("\nCheck Langfuse UI: Datasets > sentiment-benchmark-v1 > Runs")
