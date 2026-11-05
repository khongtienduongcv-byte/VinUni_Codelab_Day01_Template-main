
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load API keys from .env
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def call_openai(
    prompt,
    model="gpt-4o-mini",
    temperature=0.7,
    top_p=1.0,
    max_tokens=1000
):
    """
    Call OpenAI API and return response,
    latency and token usage.
    """

    start = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )

    latency = time.time() - start

    result = response.choices[0].message.content

    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    return {
        "response": result,
        "latency": latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }


if __name__ == "__main__":

    result = call_openai(
        prompt="Explain what Artificial Intelligence is.",
        model="gpt-4o-mini"
    )

    print("\n===== OPENAI RESULT =====")
    print("Response:", result["response"])
    print("Latency:", round(result["latency"], 3), "seconds")
    print("Input tokens:", result["input_tokens"])
    print("Output tokens:", result["output_tokens"])
