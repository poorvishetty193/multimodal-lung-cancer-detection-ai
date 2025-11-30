# Multimodal Lung Cancer Detection — Multi-Agent AI System

A production-grade multi-modal cancer diagnosis pipeline using CT scans, X-ray images, audio signals, and clinical metadata.

## Core Concept & Value

Early lung cancer diagnosis saves lives, but traditional diagnosis depends on:

- CT scan interpretation
- Symptom-based history
- Radiologist experience
- Long wait times
- High expertise requirement

Our system uses a multi-modal AI diagnostic pipeline combining:

- CT / DICOM / NIfTI
- X-Ray / PNG / JPG classifier
- Audio signals (breathing/speech patterns)
- Clinical metadata (age, smoking, symptoms)
- Fusion AI model

This produces:

- A unified cancer score
- Type prediction (Adenocarcinoma, Squamous, Large Cell, Small Cell)
- Nodule detection (CT)
- Confidence heatmaps
- Risk explanation

## The Pitch

### Problem

Lung cancer is often detected too late. Traditional diagnosis suffers from:

- Shortage of radiologists
- Manual, slow CT scan evaluation
- Fragmented data (CT, symptoms, audio)
- High rate of missed nodules
- No unified scoring

### Solution

A multi-agent AI platform that automatically processes:

- CT scan
- Image / X-ray
- Audio diagnosis
- Metadata reasoning
- AI fusion engine

### Value

- Faster diagnosis
- Reduces radiologist workload
- Gives consistent high-accuracy predictions
- Fully automated cloud pipeline
- Works with multiple modalities (CT, audio, image)
- Pause/resume long-running operations
- Agent-based scalable architecture

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

## Final Note

Your system is now a multi-agent, multi-modal cancer detection platform with:

- Real trained X-ray model
- Multi-service architecture
- CT / Audio / Metadata / Fusion
- Worker queue
- API + Agent Controller
- Docker microservices
- Storage + Redis + Logging

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
