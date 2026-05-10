# NBA Win Predictor

**Distributed Systems for Data Science — New College of Florida — Spring 2026**

A full end-to-end distributed data pipeline that predicts NBA game outcomes using machine learning.

**Live App:** https://frontend-green-iota-98.vercel.app  
**API:** http://54.147.89.206:8000  
**Author:** Henrique Furlan

---

## What It Predicts
Given two NBA teams, the model predicts which team wins and the win probability, based on each team's recent performance (last 10 games).

## Pipeline Architecture
nba_api → AWS Lambda → S3 (Bronze) → Databricks PySpark (Silver/Gold) → XGBoost/MLflow → FastAPI (EC2) → React (Vercel)

- **Ingestion:** AWS Lambda pulls game data from balldontlie API daily → stores raw JSON in S3
- **Bronze:** Raw JSON parsed into Delta table in Databricks
- **Silver:** PySpark cleans, deduplicates, adds `home_team_won` label
- **Gold:** Feature engineering — rolling 10-game win%, avg points scored/allowed, days rest
- **ML:** XGBoost classifier trained on Gold table, tracked in MLflow (Databricks)
- **Serving:** FastAPI on EC2 loads model from S3, exposes `/predict` endpoint
- **Frontend:** React app on Vercel, auto-loads team stats and displays win probability gauges

## Model Performance
- **Accuracy:** 56.3%
- **AUC-ROC:** 0.564
- **Algorithm:** XGBoost binary classifier

## How to Run the Test Script
```bash
pip install requests
python test_project.py
```

## Repo Structure
nba-win-predictor/
├── ingest.py                  # Local ingestion script
├── lambda/                    # Lambda deployment package
│   └── lambda_function.py
├── notebooks/                 # Databricks notebooks
│   ├── 01_bronze.py
│   └── 02_train_model.py
├── frontend/                  # React frontend
└── test_project.py            # Test script