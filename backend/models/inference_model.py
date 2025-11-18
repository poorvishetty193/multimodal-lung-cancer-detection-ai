from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, Any

class InferenceRecord(BaseModel):
    ct_score: float
    audio_score: float
    fusion_score: float
    metadata: Dict[str, Any]
    report_text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
