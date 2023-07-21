"""
main code for FastAPI setup
"""
import uvicorn
from fastapi import FastAPI, HTTPException
from models.models import AppDetails, GetXRayRequest, Drug, UserPrompt
from api.api import Api

description = """
API for serving as a chatbot for healthcare assistance🚀
"""

tags_metadata = [
    {
        "name": "default",
        "description": "endpoints for details of app",
    },
    {
        "name": "X-Rays",
        "description": "endpoints for x-ray interpretation",
    },
]

app = FastAPI(
    title="Baymax ChatAI",
    description=description,
    version="0.1",
    docs_url="/docs",
)


@app.get(
    "/",
)
def root():
    return {
        "message": "baymax-chatai using Fast API in Python. Go to <IP>:8000/docs for API-explorer.",
        "errors": None,
    }


@app.post("/predict", tags=["X-Rays"])
def predict(payload: GetXRayRequest):
    if _response := Api().get_xray_reports(payload.url).get("preds"):
        return {"predictions": _response}
    else:
        raise HTTPException(status_code=400, detail="Error")


@app.post("/generate")
def predict(payload: UserPrompt):
    if _response := Api().generate(payload.prompt):
        return {"response": _response}
    else:
        raise HTTPException(status_code=400, detail="Error")
