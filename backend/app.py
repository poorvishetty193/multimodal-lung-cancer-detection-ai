# backend/app.py
from fastapi import FastAPI
from backend.routes.reports import router as report_router

app = FastAPI(title="Lung Cancer AI Backend")

app.include_router(report_router)
