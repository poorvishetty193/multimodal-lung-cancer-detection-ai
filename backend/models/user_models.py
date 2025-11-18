from pydantic import BaseModel

class CreatePatient(BaseModel):
    name: str
    age: int
    sex: str

class PatientResponse(BaseModel):
    id: str
    name: str
    age: int
    sex: str
