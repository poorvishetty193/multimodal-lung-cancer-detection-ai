# Multimodal Lung Cancer Detection AI  
### CT Scans • Cough & Breath Audio • Metadata • Multi-Agent System • Google Gemini

![License](https://img.shields.io/badge/License-MIT-blue.svg)

This project is a **next-generation multimodal AI system** for early **lung cancer risk detection**.  
It integrates **CT scan analysis**, **cough & breath audio classification**, and **patient metadata** using a **multi-agent architecture**.  
The system also includes a **Google Gemini–powered Report Agent** that generates structured, clinical-style radiology reports.

> ⚠ **Educational & Research Use Only — Not a Medical Device**

---

# 🚀 Features

### 🧠 Multimodal AI
- 3D CT scan preprocessing, lung segmentation & nodule detection  
- Cough & breath audio anomaly classification (CRNN model)  
- Metadata-based risk modeling (age, smoking history, symptoms)  
- Fusion model that combines all signals for final cancer-risk scoring  

### 🤖 Multi-Agent System
- **CT-Agent** → handles CT model inference  
- **Audio-Agent** → processes cough/breath sound  
- **Metadata-Agent** → interprets patient metadata  
- **Fusion-Agent** → combines all embeddings + outputs risk score  
- **Report-Agent** → uses *Google Gemini* to generate clinical-style reports  

### 🔧 Advanced Architecture
- Tools: Gemini API, search grounding (optional), code execution, memory bank  
- Sessions & long-term memory  
- Context compaction for LLM efficiency  
- Observability (logging, metrics, tracing)  
- A2A protocol (agent-to-agent communication)  
- Evaluation pipelines for each model  

### 🌐 Full-Stack Application  
- **Frontend:** React (CT uploader + audio recorder + dashboard)  
- **Backend:** FastAPI (manages agents, sessions, ML-service requests)  
- **ML-Service:** Python microservice that runs all ML pipelines  
- **Deployment:** Docker, docker-compose, optional Kubernetes  

---

# 🧬 System Architecture

sql

             ┌──────────────────┐
             │    FRONTEND      │
             │  React Web App   │
             └─────────┬────────┘
                       │ API Calls
                       ▼
           ┌────────────────────────┐
           │     FASTAPI BACKEND    │
           └─────────┬──────────────┘
                     │ A2A Messages
                     ▼
    ┌───────────────────────────────────────────┐
    │              MULTI-AGENT SYSTEM           │
    │───────────────────────────────────────────│
    │ CT-Agent       Audio-Agent      Metadata-Agent │
    │ Fusion-Agent   Gemini Report-Agent            │
    └──────────────────┬────────────────────────────┘
                        │
                        ▼
               ┌───────────────────┐
               │    ML-SERVICE     │
               │ CT | Audio | Fusion Models │
               └───────────────────┘
---

## 📁 Folder Structure

```
cancer-detection-multimodal/
│
├── ml-service/
│   ├── ct_pipeline/
│   ├── audio_pipeline/
│   ├── metadata_pipeline/
│   ├── fusion_model/
│   ├── report_generator/
│   ├── models/
│   ├── datasets/
│   └── main_inference.py
│
├── multi-agent-system/
│   ├── agents/
│   ├── tools/
│   ├── state/
│   ├── observability/
│   └── a2a_protocol/
│
├── backend/       # FastAPI
├── frontend/      # React
├── evaluation/
└── deployment/
```


---

# 🛠 Tech Stack

### **Machine Learning**
- PyTorch  
- MONAI (medical imaging)  
- librosa / torchaudio  
- Scikit-learn  

### **LLM Tools**
- **Google Gemini API** (report generation + reasoning)
- Search grounding (optional)
- Custom memory bank  
- Context compaction  

### **Backend**
- FastAPI  
- Pydantic  
- Python A2A protocol  
- Observability stack (logs, metrics)

### **Frontend**
- React  
- TailwindCSS  
- Axios  
- Audio recorder API  

### **Deployment**
- Docker / Docker Compose  
- Optional: Kubernetes, GCP free-tier  

---

# 🔌 Google Gemini Integration

The **Report-Agent** uses Gemini to generate:
- Radiology-style CT findings  
- Audio abnormality summary  
- Combined assessment  
- Recommendations  

Example prompt:

Given the following multimodal results:
CT summary: {ct_summary}
Audio analysis: {audio_summary}
Metadata: {metadata}
Fusion risk score: {risk}

Generate a clinical-style radiology report
with findings, impression, and recommendations.


---

# 🧪 Running the Project

### 1. ML-Service Setup
cd ml-service
pip install -r requirements.txt
python main_inference.py


### 2. Backend
cd backend
uvicorn app.main:app --reload


### 3. Frontend
cd frontend
npm install
npm run dev


### 4. Docker (optional)
docker-compose up --build


---

# 🔍 Evaluation
Evaluation notebooks are in:

/evaluation/

ct_evaluation.ipynb

audio_evaluation.ipynb

fusion_evaluation.ipynb

agent_evaluation_plan.md


Metrics include:
- CT Dice, F1  
- Audio AUC, recall  
- Fusion AUC, calibration  
- System-level agent evaluation  

---

# 🧷 License

This project is released under the **MIT License**.  
See **LICENSE** file for details.

---

# ⚠ Disclaimer

This system is built **only for educational and research purposes**.  
It is **not** certified for clinical or diagnostic use.

---

# 🙌 Contributing
Pull requests, feature suggestions, and improvements are welcome!

---

# 📩 Contact
For collaborations or questions, create an issue or reach out via GitHub.
