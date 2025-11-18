from backend.db.mongo import inference_collection
from datetime import datetime

def save_inference(ct_score, audio_score, fusion_score, metadata, report):
    doc = {
        "ct_score": ct_score,
        "audio_score": audio_score,
        "fusion_score": fusion_score,
        "metadata": metadata,
        "report": report,
        "timestamp": datetime.utcnow()
    }

    inference_collection.insert_one(doc)
    return doc["_id"]
