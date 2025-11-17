"""
main_inference.py
Generates a clinical-style multimodal report by combining:
 - CT inference output
 - Audio inference output
 - Metadata (age, sex, smoking_history, symptoms)
 - Fusion score

This module depends on the Gemini REST integration available in:
    multi_agent_system.tools.google_gemini_tool.GeminiTool
(or fallback to multi_agent_system.tools.gemini_rest.generate_text)

It writes a timestamped TXT report to report_generator/outputs/.
"""

from __future__ import annotations
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Try to import the higher-level Gemini tool first (preferred)
try:
    from multi_agent_system.tools.google_gemini_tool import GeminiTool  # class-based wrapper
    _HAS_GEMINI_TOOL = True
except Exception:
    _HAS_GEMINI_TOOL = False

# Fallback: if there's a simple gemini_rest module with generate_text()
if not _HAS_GEMINI_TOOL:
    try:
        from multi_agent_system.tools.gemini_rest import generate_text  # function
        _HAS_GEMINI_REST = True
    except Exception:
        _HAS_GEMINI_REST = False
else:
    _HAS_GEMINI_REST = False


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class ReportGenerator:
    def __init__(self, gemini_model: Optional[str] = None):
        """
        If using GeminiTool class, it will be constructed automatically.
        If using gemini_rest.generate_text fallback, gemini_model is ignored.
        """
        self.gemini = None
        if _HAS_GEMINI_TOOL:
            # If google_gemini_tool.GeminiTool expects a model name, you can pass it here.
            if gemini_model:
                self.gemini = GeminiTool(model=gemini_model)
            else:
                self.gemini = GeminiTool()
        elif _HAS_GEMINI_REST:
            # using function-based fallback
            self.generate_text_fn = generate_text
        else:
            # no Gemini available: keep placeholder behavior
            self.generate_text_fn = None

    def _assemble_prompt(self, ct_result: Dict[str, Any], audio_result: Dict[str, Any],
                         metadata: Dict[str, Any], fusion_score: float) -> str:
        """
        Compose a contextual prompt for the LLM to produce a radiology-style report.
        Keep it structured and conservative (important for medical outputs).
        """
        # Pretty JSON snippets for context
        ct_json = json.dumps(ct_result, indent=2, default=str)
        audio_json = json.dumps(audio_result, indent=2, default=str)
        metadata_json = json.dumps(metadata, indent=2, default=str)

        prompt = f"""
You are a radiology assistant. Generate a concise clinical-style chest CT report combining imaging findings, audio screening,
and patient metadata. Use conservative, non-assertive language. Provide three sections: FINDINGS, IMPRESSION, and RECOMMENDATIONS.
Include a short confidence statement and suggested follow-up.

Context (do NOT invent facts; present only what's in the context):

--- CT Inference Output (JSON) ---
{ct_json}

--- Audio Inference Output (JSON) ---
{audio_json}

--- Patient Metadata (JSON) ---
{metadata_json}

--- Multimodal Fusion Score ---
Fusion risk score: {fusion_score:.3f}

Instructions:
- Under FINDINGS: summarize relevant imaging observations (appearance, segmentation, nodules if present, distribution).
- Under IMPRESSION: give a short clinical impression with risk level (low/indeterminate/high) and include the fusion probability.
- Under RECOMMENDATIONS: list 2 practical next steps (follow-up imaging timing, specialist referral, further testing).
- At the end, include a one-line CONFIDENCE/CAVEATS statement.
- Keep the report to ~6-12 sentences total, bulleted or short paragraphs.

Produce plain text only (no JSON), with section headers exactly: FINDINGS, IMPRESSION, RECOMMENDATIONS, CONFIDENCE.
"""
        return prompt.strip()

    def generate_report(self, ct_result: Dict[str, Any], audio_result: Dict[str, Any],
                        metadata: Dict[str, Any], fusion_score: float,
                        save: bool = True) -> str:
        """
        Generate the report text. Returns the text and optionally saves it to outputs/.
        """
        prompt = self._assemble_prompt(ct_result, audio_result, metadata, fusion_score)

        # Call Gemini via the available integration
        if _HAS_GEMINI_TOOL and self.gemini:
            try:
                text = self.gemini.generate_report(prompt)
            except Exception as e:
                text = f"[GEMINI ERROR] {e}\n\nPrompt:\n{prompt[:400]}"
        elif _HAS_GEMINI_REST and getattr(self, "generate_text_fn", None):
            try:
                text = self.generate_text_fn(prompt)
            except Exception as e:
                text = f"[GEMINI ERROR] {e}\n\nPrompt:\n{prompt[:400]}"
        else:
            # No Gemini — use offline placeholder
            text = "[GEMINI-PLACEHOLDER] Unable to call Gemini. Here is a conservative placeholder report.\n\n"
            text += "FINDINGS:\n- CT and audio outputs present but Gemini not active.\n\n"
            text += "IMPRESSION:\n- Indeterminate. Fusion score: {:.3f}.\n\n".format(fusion_score)
            text += "RECOMMENDATIONS:\n- Clinical correlation and follow-up imaging as indicated.\n\n"
            text += "CONFIDENCE:\n- Placeholder; Gemini not configured."

        if save:
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            fname = os.path.join(OUTPUT_DIR, f"report_{ts}.txt")
            try:
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception:
                # best-effort fallback: write to current directory
                fname = f"report_{ts}.txt"
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(text)

        return text


# Quick CLI/test helper
def quick_test():
    """
    Minimal test runner to verify integration locally.
    """
    rg = ReportGenerator()
    ct_result = {
        "cancer_risk": 0.509,
        "lung_mask_pred_shape": [64, 512, 512],
        "preprocessed_ct_shape": [64, 512, 512]
    }
    audio_result = {"audio_score": 0.51, "notes": "sample dummy audio inference"}
    metadata = {"age": 60, "sex": "male", "smoking_history": 2, "symptoms": ["cough", "short_breath"]}
    fusion_score = 0.501

    report = rg.generate_report(ct_result, audio_result, metadata, fusion_score)
    print("Generated report:\n")
    print(report)


if __name__ == "__main__":
    quick_test()
