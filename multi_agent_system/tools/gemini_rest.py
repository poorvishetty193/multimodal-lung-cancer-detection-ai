import os
import requests

API_KEY = os.getenv("GEMINI_API_KEY")

# Use a working model from your model list:
MODEL_NAME = "models/gemini-flash-latest"

BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_NAME}:generateContent"


def generate_text(prompt: str) -> str:
    """Send a prompt to Gemini via REST API."""
    if not API_KEY:
        return "[ERROR] GEMINI_API_KEY not found in environment variables."

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    url = f"{BASE_URL}?key={API_KEY}"

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(
            f"Gemini API error {response.status_code}:\n{response.text}"
        )

    data = response.json()

    # Extract text from the response
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return str(data)
