from fastapi import APIRouter
import numpy as np
import xgboost as xgb
from sklearn.metrics.pairwise import cosine_similarity

from src.api.models import MatchingRequest
from src.api.main import sbert_model, xgb_model

router = APIRouter()


@router.post("/matching", tags=["matching"])
def matching(request: MatchingRequest):
    resume_text = request.cv_text
    job_text = request.job_description

    resume_emb = sbert_model.encode(resume_text)
    job_emb = sbert_model.encode(job_text)

    cosine_sim = cosine_similarity(resume_emb.reshape(1, -1), job_emb.reshape(1, -1))

    diff = np.abs(resume_emb - job_emb)
    features = np.hstack([cosine_sim.flatten(), diff])
    features = features.reshape(1, -1)

    dmatrix = xgb.DMatrix(features)
    match_score = float(xgb_model.predict(dmatrix)[0])

    return {"match_score": match_score}
