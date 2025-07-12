
# 🧠 End-to-End MLOps Pipeline

![GitHub repo size](https://img.shields.io/github/repo-size/Bloodwingv2/End-to-End-MLOps-Pipeline)
![GitHub last commit](https://img.shields.io/github/last-commit/Bloodwingv2/End-to-End-MLOps-Pipeline)
![GitHub issues](https://img.shields.io/github/issues/Bloodwingv2/End-to-End-MLOps-Pipeline)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Bloodwingv2/End-to-End-MLOps-Pipeline)
![GitHub license](https://img.shields.io/github/license/Bloodwingv2/End-to-End-MLOps-Pipeline)

A complete MLOps workflow integrating machine learning models for health diagnostics with a CI/CD pipeline, containerization, model tracking, and a front-end dashboard. This project enables robust, automated deployment using FastAPI, Streamlit, Docker, GitHub Actions, and MLflow.

---

## 🚀 Features

- **CI/CD Automation**
  - GitHub Actions triggers every 12 hours
  - Auto-deployment using `deploy.yml` workflow
  - Separate testing workflow (`deploy.test`)

- **Model Development & Logging**
  - Models for **Diabetes**, **Heart Disease**, and **Parkinson’s** prediction
  - Integrated **SMOTE** for imbalance correction
  - Applied **Scalers** and **SHAP** for model interpretability
  - Registered models and scalers using **MLflow**
  - Dataset logging via `mlflow.data` module

- **FastAPI Backend**
  - Modular code structure
  - Added multipart form reading
  - Pre-wakeup code for cold-start reduction
  - Circular import fixes and renaming of `fastapi.py` to `main.py`

- **Streamlit Frontend**
  - Responsive UI with real-time predictions
  - Separate requirements for Streamlit and Notebooks
  - Added `metrics.json` visualization and pagination support

- **Containerization**
  - Dockerfile for backend deployment
  - Dev container support for development environments

---

## 🧰 Tech Stack

- **Languages**: Python  
- **Frameworks**: FastAPI, Streamlit  
- **ML Libraries**: XGBoost, Scikit-learn, SHAP, Imbalanced-learn (SMOTE)  
- **MLOps Tools**: MLflow, Docker, GitHub Actions  
- **CI/CD**: GitHub Workflow with custom triggers  
- **Deployment**: Render backend integration  

---

## 📁 Directory Structure

```
📦 End-to-End-MLOps-Pipeline
├── app/                 # FastAPI backend
├── frontend/            # Streamlit UI
├── models/              # ML models & scalers
├── data/                # Datasets
├── notebooks/           # Jupyter notebooks
├── .github/workflows/   # CI/CD pipelines
├── Dockerfile           # For backend container
├── requirements.txt     # Python dependencies
├── metrics.json         # Frontend visual metrics
└── README.md            # You are here
```

---

## ⚙️ Setup & Deployment

### 1. Clone the Repository
```bash
git clone https://github.com/Bloodwingv2/End-to-End-MLOps-Pipeline.git
cd End-to-End-MLOps-Pipeline
```

### 2. Set Up Environment
```bash
pip install -r requirements.txt
```

> Optional: Use `requirements_streamlit.txt` or `requirements_notebook.txt` depending on use.

### 3. Run Backend
```bash
cd app
uvicorn main:app --reload
```

### 4. Run Frontend
```bash
cd frontend
streamlit run app.py
```

---

## 📊 MLflow Dashboard

All experiments, metrics, datasets, and models are tracked using MLflow. You can run the tracking server or use a local UI.

```bash
mlflow ui
```

---

## 🖼️ Screenshots

### 🧮Dashboard UI (Streamlit)

<img width="1905" height="858" alt="output 1" src="https://github.com/user-attachments/assets/bd707b3c-3477-496d-9e9f-30cb07ae8856" />

<img width="1912" height="857" alt="output 2" src="https://github.com/user-attachments/assets/d1984208-c99f-4dbd-8551-1c13f9cf2678" />


### 🧮Dashboard UI (MLFLOW) 

### 🚦 MLflow Dashboard

<img width="1912" height="816" alt="mlflow ui" src="https://github.com/user-attachments/assets/f695b5d3-8ddd-4d58-807f-1f9aef17245c" />

### 📈 Heart Disease (MLfLow)

<img width="1473" height="477" alt="Heart_disease" src="https://github.com/user-attachments/assets/641f766e-42c3-4cb6-913c-289d8a9f28c6" />

### 📈 Parkinsons (MLfLow)

<img width="1910" height="866" alt="parkinsons" src="https://github.com/user-attachments/assets/efdb2d9b-d2dc-4016-af0f-a263e00a3f86" />

### 📈 Diabetes (MLfLow)

<img width="1907" height="858" alt="Screenshot 2025-06-13 195112" src="https://github.com/user-attachments/assets/0bace105-21f4-43bb-8390-cf7c421ec58b" />

---

### 📘 Model Staging 

<img width="1892" height="843" alt="Prod" src="https://github.com/user-attachments/assets/c590f95e-0fe4-4b52-bf04-e6cf85d43181" />

---

## 📅 Changelog Highlights

- **July 2025**  
  🔁 Renamed workflows for clarity  
  🛠 Fixed workflow syntax  
  ✅ CI/CD enhancements for production  

- **June 2025**  
  📦 Added Docker support and pre-warm backend  
  📊 Added SHAP and SMOTE logging  
  📈 Introduced dataset tracking with MLflow  
  🧠 Added XGBoost and fixed circular imports  

---

## 🤝 Contributors

👤 [Bloodwingv2](https://github.com/Bloodwingv2)

---
