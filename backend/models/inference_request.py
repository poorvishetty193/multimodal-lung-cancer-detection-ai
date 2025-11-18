from pydantic import BaseModel


class InferenceRequest(BaseModel):
    ct_image_path: str
    audio_path: str
    age: int
    sex: str
