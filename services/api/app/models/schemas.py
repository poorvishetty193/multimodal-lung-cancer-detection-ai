from pydantic import BaseModel
from typing import Optional, List, Dict

class JobCreateResponse(BaseModel):
    job_id: str
    status: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: Optional[float] = 0.0
    results: Optional[Dict] = None

class MetadataModel(BaseModel):
    patient_id: Optional[str]
    age: Optional[int]
    sex: Optional[str]
    smoking_history_pack_years: Optional[float]
    symptoms: Optional[List[str]] = []
