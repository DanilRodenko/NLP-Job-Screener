from fastapi import APIRouter
import torch

from src.api.models import ClassifierRequest
from src.api.main import model, tokenizer, device, label_encoder

router = APIRouter()

@router.post("/classify")
def classify(request: ClassifierRequest):
    encoding = tokenizer(
        request.job_description,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        category = label_encoder.inverse_transform([predicted_class])[0]

    return {"category": category}