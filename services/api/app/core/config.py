import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "minio:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minio")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY")
    STORAGE_BUCKET: str = os.getenv("STORAGE_BUCKET", "uploads")
    CELERY_BROKER: str = os.getenv("CELERY_BROKER", "amqp://guest:guest@rabbitmq:5672//")
    CELERY_BACKEND: str = os.getenv("CELERY_BACKEND", "redis://redis:6379/0")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:example@postgres:5432/lcdb")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/1")
    ML_CT_URL: str = os.getenv("ML_CT_URL", "http://ml_ct:8101/predict")
    ML_AUDIO_URL: str = os.getenv("ML_AUDIO_URL", "http://ml_audio:8102/predict")
    ML_META_URL: str = os.getenv("ML_META_URL", "http://ml_meta:8103/predict")
    ML_FUSION_URL: str = os.getenv("ML_FUSION_URL", "http://ml_fusion:8104/predict")
    class Config:
        env_file = ".env"

settings = Settings()
