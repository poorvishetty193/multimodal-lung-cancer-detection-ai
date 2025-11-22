from celery import Celery
import os

broker = os.getenv("CELERY_BROKER", "amqp://guest:guest@rabbitmq:5672//")
backend = os.getenv("CELERY_BACKEND", "redis://redis:6379/0")

celery_app = Celery("lc_tasks", broker=broker, backend=backend)
celery_app.conf.task_routes = {"tasks.process_job": {"queue": "jobs"}}
celery_app.conf.task_serializer = "json"
