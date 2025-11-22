# Lung Cancer Detection System - Starter Repo

This repo contains a complete starter full-stack project for a multimodal lung cancer detection pipeline with:
- FastAPI backend (upload API, job management)
- Celery + RabbitMQ worker for long-running jobs (pause/resume)
- Mock ML microservices: CT, audio, metadata, fusion (HTTP)
- MinIO S3-compatible object store
- Postgres, Redis for state & sessions
- Observability/logging hooks (placeholders)

This starter focuses on orchestration & integration. Replace the ML service stubs with your real models (PyTorch/nnDetection/etc.) later.

## Quick start (local)
1. Ensure Docker and docker-compose are installed.
2. From repo root:
   ```bash
   docker-compose up --build
