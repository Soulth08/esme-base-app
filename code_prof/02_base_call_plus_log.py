from dotenv import load_dotenv
from groq import Groq
from langfuse import observe, get_client

load_dotenv()

groq_client = Groq()
langfuse = get_client()

@observe(name="simple_call", as_type="generation")
def simple_call() -> dict:

    get_client().update_current_trace(
        metadata={"type": "simple_call"}
    )

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are an expert chef with 15 years of experience working in Michelin star restaurants. Your task is to help user create tasty and easy to prepare reciepies."
            },
            {"role": "user", "content": "Give me 3 meals to prepare for diner."}
        ],
        temperature=0.5
    )

    return response.choices[0].message.content


simple_call()
langfuse.flush()