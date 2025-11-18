from fastapi import FastAPI
from backend.routes.inference_routes import router as inference_router
from backend.routes.user_routes import router as user_router

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Backend running!"}

app.include_router(inference_router, prefix="/inference", tags=["Inference"])
app.include_router(user_router, prefix="/users", tags=["Users"])
