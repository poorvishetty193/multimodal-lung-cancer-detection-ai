# backend/utils/run_pipeline.py

from ml_service.report_generator.main_inference import ReportGenerator

rg = ReportGenerator()

def run_full_analysis(ct, audio, metadata, fusion_score):
    report_text = rg.generate_report(ct, audio, metadata, fusion_score)

    return report_text
