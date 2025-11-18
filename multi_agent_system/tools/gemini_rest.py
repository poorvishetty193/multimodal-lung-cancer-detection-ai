# multi_agent_system/tools/gemini_rest.py

import os
import requests

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
MODEL = "gemini-2.5-flash"   # WORKING MODEL FROM LIST

def gemini_generate(prompt: str) -> str:
    """
    Generate text via Gemini REST API.
    Returns placeholder text if API call fails.
    """

    if not GEMINI_API_KEY:
        return "[GEMINI-PLACEHOLDER] (NO API KEY) " + prompt[:200]

    url = f"{BASE_URL}/{MODEL}:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    try:
        res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})

        if res.status_code != 200:
            return f"[GEMINI-PLACEHOLDER] (HTTP {res.status_code}) {prompt[:200]}"

        data = res.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        return f"[GEMINI-PLACEHOLDER] ERROR: {str(e)}"
