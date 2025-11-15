# multi_agent_system/tools/google_gemini_tool.py

import os
import logging
from typing import Optional

import google.generativeai as genai

logger = logging.getLogger(__name__)


class GeminiTool:
    """
    Real Gemini API integration.
    Uses google-generativeai SDK.
    Set your API key in environment:
        GEMINI_API_KEY="xxxxxxx"
    """

    def __init__(self, api_key_env_var: str = "GEMINI_API_KEY", model: str = "gemini-pro"):
        self.api_key_env_var = api_key_env_var
        self.model = model

        api_key = os.environ.get(api_key_env_var)
        if not api_key:
            logger.warning("Gemini API key not set in %s. Using placeholder text.", api_key_env_var)
            self.client = None
        else:
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)

    def generate_report(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Generate a radiology-style report using Gemini LLM.
        Falls back to placeholder if no API key is present.
        """

        # Fallback (safe, offline)
        if not self.client:
            return "[GEMINI-PLACEHOLDER] " + prompt[:400] + "..."

        try:
            response = self.client.generate_content(
                prompt,
                generation_config={"max_output_tokens": max_tokens}
            )

            # SDK returns structured object → extract text
            if hasattr(response, "text"):
                return response.text

            # Some SDK versions return candidates
            return response.candidates[0].content.parts[0].text

        except Exception as e:
            logger.error("Gemini API error: %s", e)
            return f"[GEMINI-ERROR] {str(e)}"
