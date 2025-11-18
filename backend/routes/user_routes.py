from fastapi import APIRouter, HTTPException
from backend.services.user_service import user_service
from backend.models.user_models import CreatePatient, PatientResponse

router = APIRouter()

@router.post("/", response_model=PatientResponse)
def create_patient(data: CreatePatient):
    pid = user_service.create_patient(data.dict())
    return { "id": pid, **data.dict() }

@router.get("/{pid}", response_model=PatientResponse)
def get_patient(pid: str):
    patient = user_service.get_patient(pid)
    if not patient:
        raise HTTPException(404, "Patient not found")
    return patient
