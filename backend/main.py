from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from model.predict import CyberbullyingPredictor


app = FastAPI(title="Cyberbullying Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, description="Text message to analyze")


MODEL_DIR = Path(__file__).resolve().parent.parent / "model" / "saved"
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
predictor: CyberbullyingPredictor | None = None


@app.on_event("startup")
def load_predictor() -> None:
    global predictor
    if (MODEL_DIR / "cyberbullying_model.joblib").exists():
        predictor = CyberbullyingPredictor(model_dir=MODEL_DIR)


if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/results")
def results_page() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "results.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": predictor is not None}


@app.get("/info")
def info() -> dict:
    if predictor is None:
        return {"model_loaded": False, "message": "Model artifacts not found. Train first with: python model/train.py"}
    return {
        "model_loaded": True,
        "model_name": predictor.model_name,
        "sample_payload": {"text": "You are doing a great job today."},
    }


@app.post("/predict")
def predict(payload: PredictRequest) -> dict:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train model first.")
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Please provide non-empty text.")
    return predictor.predict_text(text)

