# NLP Job Screener

An AI-powered job screening system that matches resumes to job descriptions and classifies job postings by category.

## Overview

NLP Job Screener is a dual-module ML system built for automated recruitment screening:

- **Matching Module** — compares a resume against a job description and returns a relevance score (0–1)
- **Classifier Module** — predicts the job category (25 classes) from a job description text

## Tech Stack

- **ML/NLP**: Sentence-BERT (all-MiniLM-L6-v2), DistilBERT, XGBoost, scikit-learn
- **Backend**: FastAPI, Uvicorn
- **Containerization**: Docker
- **Data**: 2,484 resumes (24 categories) + 123,849 LinkedIn job postings

## Models

### Matching Module
- SBERT encodes resume and job description into embeddings
- Cosine similarity + element-wise embedding difference → feature vector
- XGBoost classifier predicts match probability
- Trained on 29,796 synthetic pairs (1:3 positive/negative ratio)

### Classifier Module
- DistilBERT fine-tuned on 103,556 LinkedIn job postings
- 25 job categories (IT, Finance, Healthcare, Engineering, etc.)
- **Accuracy: 98%**, weighted F1: 0.98

## API Endpoints

### POST `/api/v1/classify`
Predicts job category from description.

```json
{
  "job_description": "We are looking for a Python developer with experience in Django and REST APIs"
}
```

Response:
```json
{
  "category": "INFORMATION-TECHNOLOGY"
}
```

### POST `/api/v1/matching`
Returns match score between resume and job description.

```json
{
  "cv_text": "Experienced Python developer with 3 years in Django, FastAPI and REST APIs",
  "job_description": "We are looking for a Python developer with experience in Django and REST APIs"
}
```

Response:
```json
{
  "match_score": 0.988
}
```

### GET `/health`
```json
{
  "status": "ok",
  "message": "NLP Job Screener is running"
}
```

## Project Structure
NLP-Job-Screener/
├── data/
│   └── processed/
├── models/
│   └── distilbert_classifier/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_Modeling_Matching.ipynb
│   ├── 04_Modeling_Classifier.ipynb
│   └── 05_Evaluation_ipynb.ipynb
├── src/
│   └── api/
│       ├── main.py
│       ├── models.py
│       └── routers/
│           ├── classifier.py
│           └── matching.py
├── Dockerfile
└── requirements.txt

## Installation

### Local

```bash
git clone https://github.com/DanilRodenko/NLP-Job-Screener.git
cd NLP-Job-Screener
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

### Docker

```bash
docker build -t nlp-job-screener .
docker run -p 8000:8000 nlp-job-screener
```

API docs available at `http://127.0.0.1:8000/docs`

## Datasets

- [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) — Kaggle
- [LinkedIn Job Postings 2023](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) — Kaggle
