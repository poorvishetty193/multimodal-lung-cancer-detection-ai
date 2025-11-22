from minio import Minio
from core.config import settings
import io, os

_minio_client = Minio(
    settings.MINIO_ENDPOINT,
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=False
)

def ensure_bucket(bucket_name: str):
    if not _minio_client.bucket_exists(bucket_name):
        _minio_client.make_bucket(bucket_name)

def upload_bytes(bucket: str, object_name: str, data: bytes, content_type="application/octet-stream"):
    ensure_bucket(bucket)
    _minio_client.put_object(bucket, object_name, io.BytesIO(data), length=len(data), content_type=content_type)

def presigned_url(bucket: str, object_name: str, expires=3600):
    return _minio_client.presigned_get_object(bucket, object_name, expires=expires)
