FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY service/ /app/service
EXPOSE 8102
CMD ["python", "service/predict_audio.py"]
