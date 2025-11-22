FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY service/ /app/service
EXPOSE 8101
CMD ["uvicorn", "service.predict_ct:app", "--host", "0.0.0.0", "--port", "8101"]

