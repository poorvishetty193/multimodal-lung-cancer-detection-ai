FROM python:3.11-slim

WORKDIR /app

# system deps for pillow / torch (slim may need libjpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# copy
COPY service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY service /app

# default model location (volume can override)
ENV IMAGE_MODEL_PATH=/app/models/image_classifier.pt
ENV HEATMAP_DIR=/app/heatmaps
ENV PORT=8110

EXPOSE 8110
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8110"]
