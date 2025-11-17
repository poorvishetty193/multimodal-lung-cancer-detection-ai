import json
from ml_service.report_generator.models.templates import REPORT_TEMPLATE
from multi_agent_system.tools.gemini_rest import generate_text

def generate_final_report(ct_score, audio_score, metadata, fusion_score):
    """
    Combines all model outputs into a structured prompt and sends it to Gemini
    """
    prompt = REPORT_TEMPLATE.format(
        ct_score=ct_score,
        audio_score=audio_score,
        age=metadata.get("age"),
        sex=metadata.get("sex"),
        smoking=metadata.get("smoking_history"),
        symptoms=", ".join(metadata.get("symptoms", [])),
        fusion_score=fusion_score
    )
    
    result = generate_text(prompt)
    return result


# Manual test
if __name__ == "__main__":
    metadata = {
        "age": 60,
        "sex": "male",
        "smoking_history": 2,
        "symptoms": ["cough", "short_breath"]
    }

    print(generate_final_report(
        ct_score=0.52,
        audio_score=0.48,
        metadata=metadata,
        fusion_score=0.50
    ))
