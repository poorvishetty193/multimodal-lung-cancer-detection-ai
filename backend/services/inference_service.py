from ml_service.ct_pipeline.ct_inference import ct_full_inference
from ml_service.audio_pipeline.audio_inference import AudioInference
from ml_service.fusion_model.fusion_model import fusion_predict
from ml_service.report_generator.main_inference import ReportGenerator

from backend.services.db_service import get_db
from bson import ObjectId

db = get_db()

class InferenceService:

    def run_full_pipeline(self, req):

        # 1) CT INFERENCE
        ct_out = ct_full_inference(req.ct_folder)

        # 2) AUDIO INFERENCE
        audio_out = AudioInference(req.audio_file)

        # 3) FUSION MODEL
        fusion_score = fusion_predict(
            ct_out["cancer_risk"],
            audio_out["audio_score"],
            audio_out["audio_embedding"]
        )

        # 4) REPORT GENERATOR
        rg = ReportGenerator()
        report = rg.generate_report(
            ct_result=ct_out,
            audio_result=audio_out,
            metadata={"age": req.age, "sex": req.sex},
            fusion_score=fusion_score
        )

        # 5) SAVE TO MONGO
        document = {
            "patient_id": req.patient_id,
            "ct_score": ct_out["cancer_risk"],
            "audio_score": audio_out["audio_score"],
            "fusion_score": fusion_score,
            "report": report
        }

        inserted = db.inference_results.insert_one(document)

        return {
            "cancer_risk": ct_out["cancer_risk"],
            "audio_score": audio_out["audio_score"],
            "fusion_score": fusion_score,
            "report_text": report,
            "record_id": str(inserted.inserted_id)
        }


inference_service = InferenceService()
