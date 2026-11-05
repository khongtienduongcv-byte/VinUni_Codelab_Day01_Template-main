
import os
from dotenv import load_dotenv
from google import genai

# Load API key
load_dotenv()

# Initialize Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL = "gemini-2.5-flash"

# Store conversation history
history = []


def streaming_chatbot(user_message):
    """
    Gemini streaming chatbot with conversation history.
    Keeps the last 3 conversation turns (6 messages).
    """

    global history

    # Add user message
    history.append({
        "role": "user",
        "parts": [{"text": user_message}]
    })

    # Keep only the latest 6 messages
    history = history[-6:]

    print("\nGemini: ", end="", flush=True)

    # Stream response
    response_stream = client.models.generate_content_stream(
        model=MODEL,
        contents=history
    )

    full_response = ""

    for chunk in response_stream:
        if chunk.text:
            print(chunk.text, end="", flush=True)
            full_response += chunk.text

    print()

    # Save assistant response
    history.append({
        "role": "model",
        "parts": [{"text": full_response}]
    })

    # Keep only 3 latest turns
    history = history[-6:]

    return full_response


if __name__ == "__main__":

    print("===== GEMINI STREAMING CHATBOT =====")
    print("Type 'exit' to stop.\n")

    while True:

        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not user_input.strip():
            continue

        streaming_chatbot(user_input)
