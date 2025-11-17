# multi_agent_system/tools/google_gemini_tool.py

import os
import logging
import requests
import json

logger = logging.getLogger(__name__)

class GeminiTool:
    """
    Gemini REST API integration using your working model:
    models/gemini-flash-latest

    Requires:
        export GEMINI_API_KEY="your_key"
    """

    def __init__(self, api_key_env_var: str = "GEMINI_API_KEY",
                 model: str = "models/gemini-flash-latest"):
        self.api_key_env_var = api_key_env_var
        self.model = model

        self.api_key = os.environ.get(api_key_env_var)
        if not self.api_key:
            logger.warning(
                "Gemini API key not set in env var %s. Gemini calls will fail.",
                api_key_env_var
            )

        self.url = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:generateContent"

    def generate_report(self, prompt: str) -> str:
        """
        Sends a text prompt to the Gemini REST API and returns the text response.
        """

        if not self.api_key:
            return "[GEMINI-PLACEHOLDER] " + prompt[:400]

        headers = {
            "Content-Type": "application/json"
        }

        body = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ]
        }

        response = requests.post(
            f"{self.url}?key={self.api_key}",
            headers=headers,
            data=json.dumps(body)
        )

        if response.status_code != 200:
            logger.error("Gemini API error: %s\n%s", response.status_code, response.text)
            return f"[GEMINI ERROR {response.status_code}] {response.text}"

        data = response.json()

        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except:
            return "[GEMINI ERROR] Unexpected API response format"
