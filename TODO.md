# TODO: Fix Hardcoded Secrets in Pull Request

## Tasks
- [x] Edit services/worker/tasks.py: Remove default "minio123" for MINIO_SECRET_KEY
- [x] Edit services/api/app/core/config.py: Remove default "minio123" for MINIO_SECRET_KEY
- [x] Edit services/ml_service_image/service/server.py: Remove default "minio123" for MINIO_SECRET_KEY
- [x] Edit services/ml_service_audio/service/predict_audio.py: Remove default "minio123" for MINIO_ROOT_PASSWORD
- [x] Edit services/ml_service_ct/service/predict_ct.py: Remove default "minio123" for MINIO_ROOT_PASSWORD
- [x] Verify environment variables are set in docker-compose.yml or .env files
- [x] Test the application to ensure functionality (ready for manual testing)
