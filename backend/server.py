from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Multimodal Lung Cancer Detection API is running!"}
