# backend/routes/reports.py

from fastapi import APIRouter
from backend.utils.run_pipeline import run_full_analysis
from backend.database import reports_collection
from backend.models.report_model import build_report_document

router = APIRouter()

@router.post("/generate-report")
def generate_report(data: dict):
    ct = data["ct"]
    audio = data["audio"]
    metadata = data["metadata"]
    fusion = data["fusion_score"]

    final_report_text = run_full_analysis(ct, audio, metadata, fusion)

    document = build_report_document(ct, audio, metadata, fusion, final_report_text)
    reports_collection.insert_one(document)

    return {
        "status": "success",
        "report_text": final_report_text,
        "id": str(document["_id"])
    }
