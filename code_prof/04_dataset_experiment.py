from dotenv import load_dotenv
from langfuse import observe, get_client, Evaluation
from groq import Groq
import json
from datetime import datetime
from typing import Callable

load_dotenv()

groq_client = Groq()


# =============================================================================
# CREATING DATASETS
# =============================================================================

def create_sentiment_dataset():

    dataset = get_client().create_dataset(
        name="sentiment-benchmark-v1",
        description="Benchmark dataset for sentiment analysis evaluation",
        metadata={
            "created_by": "lecture_demo",
            "domain": "product_reviews",
            "version": "1.0"
        }
    )

    test_cases = [
        {
            "input": {"text": "This product is absolutely amazing! Best purchase ever!"},
            "expected_output": {"sentiment": "positive", "confidence_min": 0.8},
            "metadata": {"category": "enthusiastic_positive"}
        },
        {
            "input": {"text": "Terrible experience. Product broke after one day."},
            "expected_output": {"sentiment": "negative", "confidence_min": 0.8},
            "metadata": {"category": "strong_negative"}
        },
        {
            "input": {"text": "It's okay, nothing special but does the job."},
            "expected_output": {"sentiment": "neutral", "confidence_min": 0.5},
            "metadata": {"category": "neutral"}
        },
        {
            "input": {"text": "The quality is good but the price is too high."},
            "expected_output": {"sentiment": "mixed", "confidence_min": 0.5},
            "metadata": {"category": "mixed_sentiment"}
        },
        {
            "input": {"text": "Exceeded expectations! Will definitely buy again."},
            "expected_output": {"sentiment": "positive", "confidence_min": 0.85},
            "metadata": {"category": "positive_with_intent"}
        },
        {
            "input": {"text": "Not what I expected. Returning it tomorrow."},
            "expected_output": {"sentiment": "negative", "confidence_min": 0.7},
            "metadata": {"category": "negative_with_action"}
        },
        {
            "input": {"text": "Average product. Works as described."},
            "expected_output": {"sentiment": "neutral", "confidence_min": 0.6},
            "metadata": {"category": "factual_neutral"}
        },
        {
            "input": {"text": "I HATE THIS!!! Worst purchase of my life!!!"},
            "expected_output": {"sentiment": "negative", "confidence_min": 0.95},
            "metadata": {"category": "extreme_negative"}
        },
    ]

    for i, case in enumerate(test_cases):
        get_client().create_dataset_item(
            dataset_name="sentiment-benchmark-v1",
            input=case["input"],
            expected_output=case["expected_output"],
            metadata=case["metadata"],
        )

    print(f"✓ Created dataset with {len(test_cases)} test cases")
    return dataset


# =============================================================================
# RUNNING EXPERIMENTS
# =============================================================================

@observe()
def sentiment_task(text: str) -> dict:
    """The function we want to test."""

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


def simple_evaluator(output: dict, expected: dict) -> dict:
    """
    Simple evaluator that checks sentiment match and confidence.
    Returns scores to be logged in Langfuse.
    """

    scores = {}

    # Check sentiment match
    if output.get("sentiment") == expected.get("sentiment"):
        scores["sentiment_match"] = 1.0
    elif output.get("sentiment") in ["mixed", expected.get("sentiment")]:
        scores["sentiment_match"] = 0.5
    else:
        scores["sentiment_match"] = 0.0

    # Check confidence threshold
    confidence = output.get("confidence", 0)
    min_confidence = expected.get("confidence_min", 0)
    scores["confidence_adequate"] = 1.0 if confidence >= min_confidence else 0.0

    # Overall score
    scores["overall"] = (scores["sentiment_match"] + scores["confidence_adequate"]) / 2

    return scores


def run_experiment_manual(
    dataset_name: str,
    task_fn: Callable,
    evaluator_fn: Callable,
    experiment_name: str,
    experiment_config: dict = None
):
    """
    Run an experiment manually - useful for understanding the flow.
    For production, use get_client().run_experiment() (shown next).
    """

    # Get dataset
    dataset = get_client().get_dataset(dataset_name)

    print(f"\nRunning experiment: {experiment_name}")
    print(f"Dataset: {dataset_name} ({len(dataset.items)} items)")
    print("-" * 50)

    results = []

    for item in dataset.items:
        # V3 pattern: use item.run() as context manager which creates a root span
        with item.run(
            run_name=experiment_name,
            run_metadata=experiment_config
        ) as root_span:
            try:
                # Run the task
                output = task_fn(item.input["text"])

                # Evaluate
                scores = evaluator_fn(output, item.expected_output)

                # Log scores to Langfuse using root_span.score_trace()
                for score_name, score_value in scores.items():
                    root_span.score_trace(
                        name=score_name,
                        value=score_value,
                        comment=f"Automated evaluation for {experiment_name}"
                    )

                results.append({
                    "item_id": item.id,
                    "output": output,
                    "scores": scores,
                    "status": "success"
                })

                print(f"  ✓ Item {item.id[:8]}: {scores}")

            except Exception as e:
                results.append({
                    "item_id": item.id,
                    "error": str(e),
                    "status": "error"
                })
                print(f"  ✗ Item {item.id[:8]}: {e}")

    # Summary
    successful = [r for r in results if r["status"] == "success"]
    if successful:
        avg_overall = sum(r["scores"]["overall"] for r in successful) / len(successful)
        print(f"\n{'=' * 50}")
        print(f"Experiment complete: {len(successful)}/{len(results)} successful")
        print(f"Average overall score: {avg_overall:.2%}")

    return results


# =============================================================================
# USING THE BUILT-IN EXPERIMENT RUNNER (Recommended)
# =============================================================================

def run_experiment_builtin():
    """
    Using Langfuse's built-in experiment runner.
    Cleaner API, handles linking automatically.
    """

    dataset = get_client().get_dataset("sentiment-benchmark-v1")

    def task(*, item) -> dict:
        return sentiment_task(item.input["text"])

    def evaluator(**kwargs) -> list:
        output = kwargs.get("output")
        expected_output = kwargs.get("expected_output")
        scores = simple_evaluator(output, expected_output)

        return [
            Evaluation(name="sentiment_match", value=scores["sentiment_match"]),
            Evaluation(name="confidence_adequate", value=scores["confidence_adequate"]),
            Evaluation(name="overall", value=scores["overall"]),
        ]

    # Run the experiment (V3 API) - data expects a list of items, not the dataset object
    results = get_client().run_experiment(
        name=f"sentiment-exp-{datetime.now().strftime('%H%M%S')}",
        data=dataset.items,
        task=task,
        evaluators=[evaluator],
        description="Testing sentiment analysis with openai/gpt-oss-120b",
        metadata={"model": "openai/gpt-oss-120b", "temperature": 0.2},
    )

    print("\n✓ Experiment complete! Check Langfuse UI for results.")
    return results


# =============================================================================
# COMPARING EXPERIMENTS
# =============================================================================

def compare_models():
    """
    Run the same dataset against different model configurations.
    """

    configs = [
        {"model": "openai/gpt-oss-120b", "temperature": 0.2},
        {"model": "openai/gpt-oss-20b", "temperature": 0.5},
        {"model": "llama-3.1-8b-instant", "temperature": 0.2},
    ]

    for config in configs:
        @observe()
        def task_with_config(text: str) -> dict:
            response = groq_client.chat.completions.create(
                model=config["model"],
                messages=[
                    {"role": "system", "content": "Analyze sentiment. Respond with JSON: {sentiment, confidence, reasoning}"},
                    {"role": "user", "content": text}
                ],
                temperature=config["temperature"]
            )
            return json.loads(response.choices[0].message.content)

        experiment_name = f"{config['model']}-t{config['temperature']}"
        print(f"\n{'='*50}")
        print(f"Running: {experiment_name}")

        run_experiment_manual(
            dataset_name="sentiment-benchmark-v1",
            task_fn=task_with_config,
            evaluator_fn=simple_evaluator,
            experiment_name=experiment_name,
            experiment_config=config
        )



if __name__ == "__main__":
    print("=" * 60)
    print("DATASETS & EXPERIMENTS DEMO")
    print("=" * 60)

    # Step 1: Create dataset (only need to run ONCE)
    print("\n--- Creating Dataset --- (un-comment bellow for first run)")
    # create_sentiment_dataset()

    # Step 2: Run single experiment
    print("\n--- Running Manual Experiment ---")
    try:
        results = run_experiment_manual(
            dataset_name="sentiment-benchmark-v1",
            task_fn=sentiment_task,
            evaluator_fn=simple_evaluator,
            experiment_name="demo-experiment",
            experiment_config={"model": "openai/gpt-oss-120b"}
        )
    except Exception as e:
        print("Note: Create dataset first")
        print(f"Error: {e}")
    
    # Step 3: Using built-in runner
    print("\n--- Running Built-in Experiment ---")
    run_experiment_builtin()


    get_client().flush()
    print("\n✓ Check Langfuse UI: Datasets > sentiment-benchmark-v1 > Runs")
