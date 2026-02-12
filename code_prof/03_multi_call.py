from dotenv import load_dotenv
from groq import Groq
from langfuse import observe, get_client, propagate_attributes
import json

load_dotenv()

groq_client = Groq()
langfuse = get_client()

@observe()
def multi_step_agent(task: str) -> dict:

    with propagate_attributes(tags=["exercise1", "multi-step-agent"]):
        get_client().update_current_trace(
            metadata={"task": task}
        )

        try:
            # Step 1: Planning
            plan = _plan_steps(task)

            # Step 2: Execute each step
            results = []
            for i, step in enumerate(plan["steps"]):
                step_result = _execute_step(step, i, context=results)
                results.append(step_result)

            # Step 3: Synthesize final answer
            final_answer = _synthesize_answer(task, results)

            return {
                "task": task,
                "plan": plan,
                "step_results": results,
                "final_answer": final_answer,
                "status": "success"
            }

        except Exception as e:
            get_client().update_current_span(
                level="ERROR",
                status_message=str(e)
            )
            return {"task": task, "status": "error", "error": str(e)}


@observe(name="planning", as_type="generation")
def _plan_steps(task: str) -> dict:
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": """Break down the task into 2-4 concrete steps.
Return JSON: {"steps": ["step1", "step2", ...], "reasoning": "why these steps"}"""
            },
            {"role": "user", "content": task}
        ],
        temperature=0.3    )

    plan = json.loads(response.choices[0].message.content)

    get_client().update_current_span(
        metadata={"num_steps": len(plan.get("steps", []))}
    )

    return plan


@observe(name="execute-step")
def _execute_step(step: str, step_index: int, context: list) -> dict:
    """Execute a single step."""

    get_client().update_current_span(
        metadata={"step_index": step_index, "step": step}
    )

    context_str = "\n".join([f"- {r['output']}" for r in context]) if context else "None"

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": f"Execute this step. Previous results:\n{context_str}"
            },
            {"role": "user", "content": step}
        ],
        temperature=0.5
    )

    return {
        "step": step,
        "step_index": step_index,
        "output": response.choices[0].message.content
    }


@observe(name="synthesis", as_type="generation")
def _synthesize_answer(task: str, results: list) -> str:
    """Synthesize final answer from all step results."""

    results_text = "\n\n".join([
        f"Step {r['step_index'] + 1} ({r['step']}):\n{r['output']}"
        for r in results
    ])

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "Synthesize a comprehensive answer from the step results."
            },
            {"role": "user", "content": f"Task: {task}\n\nResults:\n{results_text}"}
        ],
        temperature=0.3
    )

    answer = response.choices[0].message.content

    return answer


print("\n--- Exercise 1: Multi-step Agent ---")
result = multi_step_agent("Explain how to set up a Python virtual environment")
print(result)
if result['status'] == 'success':
    print(f"Plan: {result['plan']}")
    print(f"Final answer: {result['final_answer'][:200]}...")
else:
    print(f"Error: {result['error']}")
