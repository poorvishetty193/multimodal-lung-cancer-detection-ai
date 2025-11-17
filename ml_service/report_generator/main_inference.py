from multi_agent_system.tools.gemini_rest import generate_text


def generate_final_report(
        ct_score: float,
        audio_score: float,
        metadata: dict,
        fusion_score: float
):
    """
    Generates a full clinical-style report using the Gemini REST API.
    """

    prompt = f"""
You are an AI health assistant helping generate a brief clinical-style report.

CT Score: {ct_score:.3f}
Audio Score: {audio_score:.3f}
Fusion Risk Score: {fusion_score:.3f}

Metadata:
Age: {metadata.get("age")}
Sex: {metadata.get("sex")}
Smoking History (0–3): {metadata.get("smoking_history")}
Symptoms: {", ".join(metadata.get("symptoms", []))}

Write a short medical-style summary including:
- Findings
- Risk interpretation
- Recommendations
- Confidence note
Keep the tone conservative and non-diagnostic.
"""

    result = generate_text(prompt)
    return result


if __name__ == "__main__":
    metadata = {
        "age": 60,
        "sex": "male",
        "smoking_history": 2,
        "symptoms": ["cough", "short_breath"]
    }

    print(
        generate_final_report(
            ct_score=0.52,
            audio_score=0.48,
            metadata=metadata,
            fusion_score=0.50
        )
    )
