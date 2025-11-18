from ml_service.ct_pipeline.ct_inference import CTInference
from ml_service.audio_pipeline.audio_inference import AudioInference
from ml_service.report_generator.main_inference import ReportGenerator
from ml_service.fusion_model.fusion_model import FusionModel

class InferenceService:

    def __init__(self):
        self.ct_model = CTInference()
        self.audio_model = AudioInference()
        self.fusion = FusionModel()
        self.report_gen = ReportGenerator()

    def run_inference(self, ct_input, audio_input, metadata):
        ct_result = self.ct_model.predict(ct_input)
        audio_result = self.audio_model.predict(audio_input)
        fusion_score = self.fusion.combine(
            ct_result.get("cancer_risk", 0),
            audio_result.get("audio_score", 0),
            metadata
        )
        report = self.report_gen.generate_report(
            ct_result,
            audio_result,
            metadata,
            fusion_score
        )

        return {
            "ct_result": ct_result,
            "audio_result": audio_result,
            "fusion_score": fusion_score,
            "report": report
        }
