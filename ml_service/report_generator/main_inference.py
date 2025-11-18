from multi_agent_system.tools.gemini_rest import gemini_generate
from ml_service.report_generator.local_report_model import generate_local_report

class ReportGenerator:

    def generate_report(self, ct_result, audio_result, metadata, fusion_score):

        prompt = f"""
You are a medical report generator. Use the following information:

CT cancer score: {ct_result.get('cancer_risk')}
Audio score: {audio_result.get('audio_score')}
Patient metadata: {metadata}
Fusion risk score: {fusion_score}

Write a radiology-style report with:
- FINDINGS
- IMPRESSION
- RECOMMENDATIONS
- CONFIDENCE

Keep language short, factual, and clinical.
"""

        # 1) Try Gemini REST
        try:
            response = gemini_generate(prompt)
            if response:
                return response
        except Exception as e:
            print("[WARNING] Gemini failed:", e)

        # 2) Fallback: Local offline model
        return generate_local_report(ct_result, audio_result, metadata, fusion_score)
