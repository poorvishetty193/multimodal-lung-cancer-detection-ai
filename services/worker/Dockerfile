FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 🔥 FIX: create logs directory
RUN mkdir -p /app/logs

COPY tasks.py /app/tasks.py
COPY orchestrator_stub.py /app/orchestrator_stub.py

ENV PYTHONUNBUFFERED=1

CMD ["python", "tasks.py"]
