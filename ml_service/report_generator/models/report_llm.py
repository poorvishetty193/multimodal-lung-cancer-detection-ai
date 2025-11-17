# ml_service/report_generator/models/report_llm.py

import os
import json
import requests

class ReportLLM:
    """
    Uses Google Gemini REST API to generate radiology-style reports.
    Falls back to template-based report if no API key is available.
    """

    def __init__(self, model_name="models/gemini-2.5-flash"):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = model_name

    # -------------------------------------------------------------
    # Main Report Method (Gemini + Fallback)
    # -------------------------------------------------------------
    def generate(self, ct_score, audio_score, metadata, fusion_score):
        prompt = self._build_prompt(ct_score, audio_score, metadata, fusion_score)

        # If no API key → offline fallback
        if not self.api_key:
            return self._fallback_report(ct_score, audio_score, metadata, fusion_score)

        return self._generate_via_gemini(prompt)

    # -------------------------------------------------------------
    # Build Prompt for Gemini
    # -------------------------------------------------------------
    def _build_prompt(self, ct_score, audio_score, metadata, fusion_score):
        return f"""
You are an expert radiologist. Create a concise clinical report.

CT cancer-risk score: {ct_score:.3f}
Audio cough score: {audio_score:.3f}
Fusion (overall) score: {fusion_score:.3f}

Patient metadata:
{json.dumps(metadata, indent=2)}

Write sections:
- Findings
- Impression
- Recommendations
Use cautious clinical language.
"""

    # -------------------------------------------------------------
    # Gemini API Call
    # -------------------------------------------------------------
    def _generate_via_gemini(self, prompt):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"

        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": prompt}]}]}

        response = requests.post(url, json=data, headers=headers)

        if response.status_code != 200:
            return f"[Gemini Error {response.status_code}] {response.text}"

        try:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except:
            return "[ERROR] Bad Gemini response format."

    # -------------------------------------------------------------
    # Offline fallback report (no Gemini)
    # -------------------------------------------------------------
    def _fallback_report(self, ct_score, audio_score, metadata, fusion_score):
        return f"""
=== Offline Auto-Generated Report ===

Findings:
CT-based model indicates probability of cancer = {ct_score:.2f}.
Audio-based model probability = {audio_score:.2f}.
Combined fusion probability = {fusion_score:.2f}.

Impression:
The results indicate a moderate but uncertain risk. Since no LLM is available,
this interpretation is simplified and should not be used for clinical decisions.

Recommendations:
- Consider clinical correlation.
- Obtain further diagnostic imaging.
- Consult a specialist.
"""  
