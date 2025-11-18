# backend/main.py
from fastapi import FastAPI
from backend.routes.inference_routes import router as inference_router

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Backend running!"}

app.include_router(inference_router)
