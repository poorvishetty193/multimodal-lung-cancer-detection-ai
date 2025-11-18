from pydantic import BaseModel

class InferenceRequest(BaseModel):
    patient_id: str
    age: int
    sex: str
    ct_folder: str
    audio_file: str

class InferenceResponse(BaseModel):
    cancer_risk: float
    audio_score: float
    fusion_score: float
    report_text: str
    record_id: str
