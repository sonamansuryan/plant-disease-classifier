import os
import sys
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import CFG
from predict import DiseasePredictor

app = FastAPI(
    title="Plant Disease Classifier",
    description=(
        "Upload a plant leaf image to classify the **disease** "
        "(plant-agnostic: e.g. 'late blight' for both tomato and potato)."
    ),
    version="1.0.0",
)

_predictor: DiseasePredictor | None = None


@app.on_event("startup")
def load_model() -> None:
    global _predictor
    ckpt = str(CFG.OUTPUT_DIR / "best_model.pth")
    _predictor = DiseasePredictor(ckpt)
    print("Model loaded and ready.")


@app.post(
    "/predict",
    summary="Classify plant disease from an image",
    response_description="Top-1 disease with confidence, plus top-3 candidates",
)
async def predict_disease(file: UploadFile = File(..., description="Plant leaf image (jpg/png)")):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    suffix = os.path.splitext(file.filename)[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = _predictor.tta_predict(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.unlink(tmp_path)

    return JSONResponse(content={
        "disease":    result["disease"],
        "confidence": round(result["confidence"], 4),
        "top3":       result["top3"],
    })


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "model_loaded": _predictor is not None}
