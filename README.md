# Multimodal Lung Cancer Detection — Multi-Agent AI System

A production-grade multi-modal cancer diagnosis pipeline using CT scans, X-ray images, audio signals, and clinical metadata.

🧩 Problem Statement

Lung cancer diagnosis traditionally depends on multiple disconnected sources: CT scans, X-ray images, patient speech biomarkers, and clinical metadata such as age, symptoms, and smoking history.
Manually interpreting all these modalities is slow, error-prone, and requires high clinical expertise.

Other pain points:

Radiologists must analyze CT slice-by-slice, increasing fatigue and error rates

Image orientation or compression often distorts patterns

Speech anomalies linked to lung obstruction are rarely used due to lack of tools

Metadata is ignored though it significantly influences cancer probability

No unified system exists to combine all modalities for an accurate, reproducible diagnosis

🎯 Solution Statement

This project introduces a fully automated multi-agent diagnostic system that processes:

CT scans (NIfTI / DICOM / ZIP)

Chest X-ray or image files (PNG/JPG)

Patient audio

Patient metadata (age, smoking pack-years, symptoms)

Each modality is handled by a specialized agent, and outputs are fused by a Fusion Agent to produce:

Cancer classification

Risk score

Reasoning (nodules, anomalies, metadata contribution)

Heatmaps or probability distributions

This creates a reliable clinical decision support system with consistent accuracy.

🌟 Core Concept & Value
Concept

A modular, scalable multi-agent diagnostic pipeline where each modality is handled by an independent ML microservice. Agents collaborate using an orchestrator to deliver final diagnosis.

Value

Accelerates diagnosis

Reduces radiologist workload

Handles any orientation / compression of images

Uses multi-modal evidence instead of single modality

Real-time diagnosis in under 10 seconds

New modalities can be added with zero changes to existing agents

🚀 The Pitch
🔥 Problem

Diagnosing lung cancer is slow, inconsistent, and highly dependent on manual interpretation of CT scans alone.

⭐ Solution

A multimodal multi-agent system that automatically interprets CT scans, images, audio biomarkers, and metadata — then fuses results into a final diagnosis.

💎 Value

Accurate, scalable AI that reduces diagnostic time, improves consistency, and integrates seamlessly into hospitals or remote diagnosis tools.

🏛 System Architecture
             ┌──────────────────────────────────────────┐
             │              Frontend (React)             │
             │ Upload CT / Image / Audio + Metadata      │
             └──────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│                              API (FastAPI)                         │
│ - Uploads files to MinIO                                          │
│ - Stores job in Redis                                             │
│ - Enqueues job                                                     │
└────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│                        Worker (Task Engine)                        │
│   Multi-Agent orchestration:                                       │
│      ├── CT Agent                                                  │
│      ├── Image Agent                                               │
│      ├── Audio Agent                                               │
│      ├── Metadata Agent                                            │
│      └── Fusion Agent                                              │
│   Uses long-running job flow (pause/resume)                       │
└────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
             ┌──────────────────────────────────────────┐
             │                Results API                │
             │            (Risk Score + Explainability)  │
             └──────────────────────────────────────────┘

## System Overview

The system uses a Multi-Agent Architecture, including:

### Sequential Agents

Each modality is processed step-by-step:

- CT Agent
- Image Agent
- Audio Agent
- Metadata Agent
- Fusion Agent

### Parallel Agents

CT and Image models can run simultaneously.

### Loop Agents

Worker continuously polls Redis queue (loop agent).

### Tools Used

| Tool                  | Used?     | Purpose                          |
|-----------------------|-----------|----------------------------------|
| LLM-powered Agent     | ❌ (planned) | Will generate reports & explanations |
| Parallel Agents       | ✔ Yes     | CT/Image can run in parallel     |
| Sequential Agents     | ✔ Yes     | Fusion depends on upstream results |
| Loop Agents           | ✔ Yes     | Worker job polling               |
| MCP                   | ❌ (future) | For future tool orchestration    |
| Custom Tools          | ✔ Yes     | Storage (MinIO), Redis, Docker services |
| Built-in Tools        | ✔ Yes     | Code execution, HTTP requests    |

## File Structure

```
multimodal-lung-cancer-detection-ai/
│
├── services/
│   ├── api/
│   │   ├── app/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── worker/
│   │   ├── tasks.py
│   │   ├── orchestrator/
│   │   │   └── agent_controller.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── ml_service_ct/
│   │   ├── service/predict_ct.py
│   │   ├── models/
│   │   └── Dockerfile
│   │
│   ├── ml_service_audio/
│   │   ├── service/predict_audio.py
│   │   └── Dockerfile
│   │
│   ├── ml_service_metadata/
│   │   ├── service/predict_meta.py
│   │   └── Dockerfile
│   │
│   ├── ml_service_image/
│   │   ├── service/train.py
│   │   ├── service/infer.py
│   │   ├── service/models.py
│   │   ├── service/utils.py
│   │   └── service/server.py
│   │
│   ├── ml_service_fusion/
│   │   ├── service/predict_fusion.py
│   │   └── Dockerfile
│
├── docker-compose.yml
├── .env
└── README.md
```

## Architecture

```
User Upload → API → Redis Queue → Worker → Agent Controller
          ↓                   ↓
     MinIO Storage ← CT / Image / Audio Files

Agent Controller → CT Model Service
                  → Image Model Service
                  → Audio Service
                  → Metadata Service
                  → Fusion Engine

Fusion Output → API → Frontend UI
```

## Technical Implementation

### Backend

- FastAPI
- Redis (state + queue)
- MinIO (storage)
- PostgreSQL (user jobs)
- Docker Microservices
- Python Agent Controller

### AI Models

- CT Model → Lung nodule detection (dummy now, can be upgraded)
- Image Model → ResNet50 classifier (trained using your dataset)
- Audio Model → Future: CNN/RNN
- Metadata Model → Rule-based (can be upgraded)
- Fusion Model → Normalized averaged probabilities

### Worker

- Loop-based agent
- Processes job queue
- Pause/Resume supported through Redis

### Observability

- Loguru logs
- worker.log
- Docker logs
- Request/Response trace

## OpenAPI Tools

All microservices expose:

- `/predict` (POST)

Auto-documented using FastAPI Swagger:

`http://localhost:<port>/docs`

## Long-Running Operations

Pause / Resume Supported

Each job stores:

- `status = queued | running | paused | completed | failed`
- `progress = 0.0 → 100.0`

Worker checks before processing:

```python
if status == "paused":
    requeue
```

## Sessions & Memory

InMemorySessionService

Stores temporary job states in Redis.

### Future: Memory Bank

LLM agent can store long-term clinical interpretation.

### Context Engineering

Metadata is compacted before passing to fusion model.

## Observability

- Logging: Loguru logs in `/app/logs/worker.log`
- Tracing: Job ID propagated through all agents
- Metrics: (Pending) Prometheus exporters

## Agent Evaluation

You can evaluate each agent independently:

- CT → nodules + probabilities
- Image → classification accuracy
- Audio → anomaly scoring
- Metadata → rule-based correctness
- Fusion → weighted probability consistency

## A2A Protocol (Agent-to-Agent)

Communication between agents uses HTTP JSON RPC style:

- CT Agent → Fusion Agent
- Image Agent → Fusion Agent
- Audio Agent → Fusion Agent
- Metadata Agent → Fusion Agent

## Deployment

### Local

```bash
docker compose build --no-cache
docker compose up
```

### Production Options

- Kubernetes
- Azure Container Apps
- AWS ECS
- Docker Swarm

## Pending Work

| Feature                                      | Status    |
|----------------------------------------------|-----------|
| Replace dummy CT model with real nodule detector | ⏳ Pending |
| Replace audio dummy model                    | ⏳ Pending |
| Add LLM-powered radiology report generator   | ⏳ Planned |
| Heatmap visualisation (Grad-CAM)             | ⏳ Pending |
| Full UI dashboard                            | ⏳ Pending |
| Authentication + sessions                    | ⏳ Pending |
| Add A2A LLM-based decision agent             | ⏳ Planned |
| Add monitoring dashboards                    | ⏳ Planned |

---

## 👩‍💻 Author

**Poorvi Shetty**
💡 Computer Science Student
📘 Full Stack + Machine Learning Developer

---

## 📝 License

This project is released under the **MIT License**.
You are free to use, modify, and distribute it for learning or research purposes.

---

### ⭐ If you like this project, give it a star on GitHub! ⭐
