
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load API keys
load_dotenv()

# Initialize Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def call_gemini(
    prompt,
    model="gemini-2.5-flash",
    temperature=0.7,
    top_p=0.95,
    max_tokens=1000
):
    """
    Call Gemini API and return response,
    latency and token usage.
    """

    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens
    )

    start = time.time()

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config
    )

    latency = time.time() - start

    result = response.text

    usage = response.usage_metadata

    input_tokens = usage.prompt_token_count
    output_tokens = usage.candidates_token_count

    return {
        "response": result,
        "latency": latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }


if __name__ == "__main__":

    result = call_gemini(
        prompt="Explain what Artificial Intelligence is.",
        model="gemini-2.5-flash"
    )

    print("\n===== GEMINI RESULT =====")
    print("Response:", result["response"])
    print("Latency:", round(result["latency"], 3), "seconds")
    print("Input tokens:", result["input_tokens"])
    print("Output tokens:", result["output_tokens"])
