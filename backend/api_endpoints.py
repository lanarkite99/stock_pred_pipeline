import asyncio
import logging
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.config import Config
from src.exception import PipelineError
from src.pipelines.inference_pipeline import predict_child, predict_parent
from src.pipelines.train_pipeline import train_child_model, train_parent_model

logger = logging.getLogger(__name__)
router = APIRouter()
config = Config()


class TickerRequest(BaseModel):
    ticker: str = Field(..., min_length=1, description="Yahoo Finance ticker, e.g. RELIANCE.NS")


def _child_model_path(ticker: str) -> str:
    return os.path.join(config.workdir, ticker.lower(), f"{ticker}_child_model.pt")


def _parent_model_path() -> str:
    return os.path.join(config.parent_dir, f"{config.parent_ticker}_parent_model.pt")


def _not_found_for_missing_artifact(exc: Exception) -> bool:
    message = str(exc).lower()
    return "missing pytorch model" in message or "missing scaler" in message or "not found" in message


@router.get("/")
def root():
    return {
        "project": "agentic-stock-pred-end2end",
        "description": "Minimal FastAPI endpoints for parent/child training and inference.",
        "features": [
            "Feast-backed ingestion",
            "Parent model training",
            "Child transfer learning",
            "Parent inference",
            "Child inference",
        ],
        "endpoints": {
            "GET /": "Project summary",
            "GET /health": "Health check",
            "POST /train-parent": "Train parent market model",
            "POST /train-child": "Train child model for a ticker",
            "POST /predict-parent": "Predict using parent model",
            "POST /predict-child": "Predict using child model for a ticker",
        },
        "quick_start": {
            "train_parent": {"method": "POST", "path": "/train-parent"},
            "train_child": {"method": "POST", "path": "/train-child", "body": {"ticker": "RELIANCE.NS"}},
            "predict_parent": {"method": "POST", "path": "/predict-parent"},
            "predict_child": {"method": "POST", "path": "/predict-child", "body": {"ticker": "RELIANCE.NS"}},
        },
    }


@router.get("/health")
def health():
    return {"status": "healthy"}


@router.post("/train-parent")
async def train_parent_endpoint():
    try:
        result = await asyncio.to_thread(train_parent_model)
        return {
            "status": "completed",
            "model_exists": os.path.exists(_parent_model_path()),
            "result": result,
        }
    except PipelineError as e:
        logger.error(f"Parent training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.exception("Unexpected parent training failure")
        raise HTTPException(status_code=500, detail="parent training failed") from e


@router.post("/train-child")
async def train_child_endpoint(request: TickerRequest):
    ticker = request.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")

    try:
        result = await asyncio.to_thread(train_child_model, ticker)
        return {
            "status": "completed",
            "model_exists": os.path.exists(_child_model_path(ticker)),
            "result": result,
        }
    except PipelineError as e:
        logger.error(f"Child training failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.exception(f"Unexpected child training failure for {ticker}")
        raise HTTPException(status_code=500, detail=f"child training failed for {ticker}") from e


@router.post("/predict-parent")
async def predict_parent_endpoint():
    try:
        result = await asyncio.to_thread(predict_parent)
        return {"status": "completed", "result": result}
    except PipelineError as e:
        status_code = 404 if _not_found_for_missing_artifact(e) else 500
        raise HTTPException(status_code=status_code, detail=str(e)) from e
    except Exception as e:
        logger.exception("Unexpected parent prediction failure")
        raise HTTPException(status_code=500, detail="parent prediction failed") from e


@router.post("/predict-child")
async def predict_child_endpoint(request: TickerRequest):
    ticker = request.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")

    try:
        result = await asyncio.to_thread(predict_child, ticker)
        return {"status": "completed", "result": result}
    except PipelineError as e:
        status_code = 404 if _not_found_for_missing_artifact(e) else 500
        raise HTTPException(status_code=status_code, detail=str(e)) from e
    except Exception as e:
        logger.exception(f"Unexpected child prediction failure for {ticker}")
        raise HTTPException(status_code=500, detail=f"child prediction failed for {ticker}") from e
