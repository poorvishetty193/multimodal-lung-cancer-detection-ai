# multi-agent-system/agents/report_agent.py
import logging
from typing import Dict, Any

from multi_agent_system.tools.google_gemini_tool import GeminiTool

logger = logging.getLogger(__name__)


class ReportAgent:
    """
    Report-Agent: composes a prompt using CT/Audio/Fusion outputs and calls Gemini via the GeminiTool.
    """

    def __init__(self, api_key_env_var: str = "GEMINI_API_KEY"):
        self.tool = GeminiTool(api_key_env_var=api_key_env_var)

    def run(self, ct_summary: str, audio_summary: str, metadata_summary: str, fusion_score: float) -> Dict[str, Any]:
        prompt = (
            "You are an assistant that generates concise clinical-style radiology reports.\n\n"
            f"CT summary:\n{ct_summary}\n\n"
            f"Audio summary:\n{audio_summary}\n\n"
            f"Metadata:\n{metadata_summary}\n\n"
            f"Fusion risk score: {fusion_score:.3f}\n\n"
            "Produce a brief report with: Findings, Impression, Recommendations. "
            "Keep conservative language and include confidence / caveats."
        )
        logger.info("ReportAgent: sending prompt to Gemini")
        output = self.tool.generate_report(prompt)
        return {
            "report_text": output,
            "provenance": {"source": "report_agent"}
        }
