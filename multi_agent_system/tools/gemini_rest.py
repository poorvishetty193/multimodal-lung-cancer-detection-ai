# multi_agent_system/tools/gemini_rest.py
import os
import requests
import json

# NEW: Pick a valid model from your list_models output
GEMINI_MODEL = "models/gemini-flash-latest"

API_KEY = os.getenv("GEMINI_API_KEY")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_MODEL}:generateContent"


def generate_gemini_report(prompt: str) -> str:
    """
    Calls Gemini via REST API and returns text.
    """

    if not API_KEY:
        return "[ERROR] GEMINI_API_KEY is missing. Set it with:  setx GEMINI_API_KEY \"YOUR_KEY\""

    headers = {
        "Content-Type": "application/json"
    }

    body = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    params = {"key": API_KEY}

    response = requests.post(API_URL, headers=headers, params=params, json=body)

    if response.status_code != 200:
        return f"[Gemini REST Error {response.status_code}]: {response.text}"

    data = response.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return f"[Gemini Parse Error] Raw response: {json.dumps(data, indent=2)}"
