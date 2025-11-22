#!/bin/bash
# Create MinIO bucket if needed - optional script
mc alias set local http://localhost:9000 minio minio123
mc mb local/uploads || true
