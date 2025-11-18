# backend/models/report_model.py

def build_report_document(ct, audio, metadata, fusion_score, final_report_text):
    return {
        "ct_result": ct,
        "audio_result": audio,
        "metadata": metadata,
        "fusion_score": fusion_score,
        "final_report": final_report_text
    }
