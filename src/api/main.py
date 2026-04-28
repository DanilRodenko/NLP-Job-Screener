import torch
import joblib
import xgboost as xgb

from fastapi import FastAPI
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer

app = FastAPI(title="NLP Job Screener")

device = torch.device("cpu")

print("Loading XGBoost...")
xgb_model = xgb.Booster()
xgb_model.load_model("models/matching_model.json")
print("XGBoost loaded")

print("Loading DistilBERT...")
tokenizer = DistilBertTokenizer.from_pretrained("models/distilbert_classifier")
model = DistilBertForSequenceClassification.from_pretrained(
    "models/distilbert_classifier"
)
model = model.to(device)
model.eval()
print("DistilBERT loaded")

print("Loading SBERT...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
print("SBERT loaded")

label_encoder = joblib.load("models/label_encoder.joblib")
print("Label encoder loaded")


@app.get("/health")
def health():
    return {"status": "ok", "message": "NLP Job Screener is running"}


from src.api.routers.classifier import router as classifier_router
from src.api.routers.matching import router as matching_router

app.include_router(classifier_router, prefix="/api/v1")
app.include_router(matching_router, prefix="/api/v1")
