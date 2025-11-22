from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.upload import router as upload_router
from routes.jobs import router as jobs_router
from core.config import settings
from core.logger import logger

app = FastAPI(title="Lung Cancer Detection API - Starter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="/api/v1/scan", tags=["scan"])
app.include_router(jobs_router, prefix="/api/v1/job", tags=["jobs"])

@app.get("/")
async def root():
    return {"status": "ok", "service": "lung-cancer-api", "version": "0.1"}
