import asyncio
import json
import os

from fastapi import APIRouter, HTTPException, Response

from backend.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    MonitorResponse,
    PredictionCompletedResponse,
    RootResponse,
    TaskStatusResponse,
    TickerRequest,
    TrainingAcceptedResponse,
)
from backend.state import (
    ANALYSIS_COUNTER,
    ANALYSIS_FAILURES,
    ANALYSIS_LATENCY,
    PREDICTION_COUNTER,
    PREDICTION_LATENCY,
)
from backend.tasks import get_or_set_cache, get_task_status_redis, run_training
from logger.logger import get_logger
from src.config import Config
from src.exception import PipelineError
from src.utils import save_json

logger = get_logger()
router = APIRouter()
config = Config()


def _child_model_path(ticker: str) -> str:
    return os.path.join(config.workdir, ticker.lower(), f"{ticker}_child_model.pt")


def _parent_model_path() -> str:
    return os.path.join(config.parent_dir, f"{config.parent_ticker}_parent_model.pt")


def _latest_analysis_path(ticker: str) -> str:
    return os.path.join(config.workdir, ticker.lower(), "latest_analysis.json")


def _latest_monitor_path(ticker: str) -> str:
    return os.path.join(config.workdir, ticker.lower(), "monitor", "latest_monitor.json")


def _load_saved_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as e:
        logger.warning("Failed to load saved json %s: %s", path, e)
        return None


def _not_found_for_missing_artifact(exc: Exception) -> bool:
    message = str(exc).lower()
    return "missing pytorch model" in message or "missing scaler" in message or "not found" in message


async def _start_auto_training(task_id: str, train_fn, chain_fn, detail: str):
    current_status = get_task_status_redis(task_id)
    if current_status and current_status.get("status") == "running":
        return {"status": "training", "detail": detail, "task_id": task_id}

    await run_training(task_id, train_fn, chain_fn=chain_fn)
    return {"status": "training", "detail": detail, "task_id": task_id}


async def _start_auto_training_with_args(task_id: str, train_fn, train_args: tuple, chain_fn, detail: str):
    current_status = get_task_status_redis(task_id)
    if current_status and current_status.get("status") == "running":
        return {"status": "training", "detail": detail, "task_id": task_id}

    await run_training(task_id, train_fn, *train_args, chain_fn=chain_fn)
    return {"status": "training", "detail": detail, "task_id": task_id}


@router.get("/", response_model=RootResponse)
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
            "GET /status/{task_id}": "Get async training task status",
            "GET /artifacts/{ticker}/analysis": "Get latest saved analysis output",
            "GET /artifacts/{ticker}/monitor": "Get latest saved monitor output",
            "POST /monitor/{ticker}": "Run monitoring checks for a ticker",
            "POST /train-parent": "Train parent market model",
            "POST /train-child": "Train child model for a ticker",
            "POST /predict-parent": "Predict using parent model",
            "POST /predict-child": "Predict using child model for a ticker",
            "POST /analyze": "Generate a minimal analysis summary for a ticker",
        },
        "quick_start": {
            "train_parent": {"method": "POST", "path": "/train-parent"},
            "train_child": {"method": "POST", "path": "/train-child", "body": {"ticker": "RELIANCE.NS"}},
            "predict_parent": {"method": "POST", "path": "/predict-parent"},
            "predict_child": {"method": "POST", "path": "/predict-child", "body": {"ticker": "RELIANCE.NS"}},
            "analyze": {"method": "POST", "path": "/analyze", "body": {"ticker": "RELIANCE.NS"}},
            "monitor": {"method": "POST", "path": "/monitor/RELIANCE.NS"},
        },
    }


@router.get("/health", response_model=HealthResponse)
def health():
    return {"status": "healthy"}


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    status = get_task_status_redis(task_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"task {task_id} not found")
    return {"task_id": task_id, **status}


@router.get("/artifacts/{ticker}/analysis")
def get_latest_analysis(ticker: str):
    ticker_u = ticker.strip().upper()
    if not ticker_u:
        raise HTTPException(status_code=400, detail="ticker is required")

    data = _load_saved_json(_latest_analysis_path(ticker_u))
    if data is None:
        raise HTTPException(status_code=404, detail=f"latest analysis not found for {ticker_u}")
    return data


@router.get("/artifacts/{ticker}/monitor")
def get_latest_monitor(ticker: str):
    ticker_u = ticker.strip().upper()
    if not ticker_u:
        raise HTTPException(status_code=400, detail="ticker is required")

    data = _load_saved_json(_latest_monitor_path(ticker_u))
    if data is None:
        raise HTTPException(status_code=404, detail=f"latest monitor not found for {ticker_u}")
    return data


@router.post("/monitor/{ticker}", response_model=MonitorResponse)
async def monitor_ticker(ticker: str):
    ticker_u = ticker.strip().upper()
    if not ticker_u:
        raise HTTPException(status_code=400, detail="ticker is required")

    try:
        from monitoring.run_monitoring import run_monitoring

        return await asyncio.to_thread(run_monitoring, ticker_u)
    except Exception as e:
        logger.exception("Monitoring failed for %s", ticker_u)
        raise HTTPException(status_code=500, detail=f"monitoring failed for {ticker_u}") from e


@router.post("/train-parent", response_model=TrainingAcceptedResponse)
async def train_parent_endpoint(response: Response):
    task_id = "parent_training"
    current_status = get_task_status_redis(task_id)
    if current_status and current_status.get("status") == "running":
        response.status_code = 202
        return {"status": "running", "task_id": task_id}

    try:
        from src.pipelines.train_pipeline import train_parent_model

        await run_training(task_id, train_parent_model)
        response.status_code = 202
        return {"status": "started", "task_id": task_id}
    except PipelineError as e:
        logger.error(f"Parent training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.exception("Unexpected parent training failure")
        raise HTTPException(status_code=500, detail="parent training failed") from e


@router.post("/train-child", response_model=TrainingAcceptedResponse)
async def train_child_endpoint(request: TickerRequest, response: Response):
    ticker = request.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")

    task_id = ticker.lower()
    current_status = get_task_status_redis(task_id)
    if current_status and current_status.get("status") == "running":
        response.status_code = 202
        return {"status": "running", "task_id": task_id}

    try:
        from src.pipelines.train_pipeline import train_child_model

        await run_training(task_id, train_child_model, ticker)
        response.status_code = 202
        return {"status": "started", "task_id": task_id}
    except PipelineError as e:
        logger.error(f"Child training failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.exception(f"Unexpected child training failure for {ticker}")
        raise HTTPException(status_code=500, detail=f"child training failed for {ticker}") from e


@router.post(
    "/predict-parent",
    response_model=PredictionCompletedResponse | TrainingAcceptedResponse,
)
async def predict_parent_endpoint(response: Response):
    try:
        from src.pipelines.inference_pipeline import predict_parent

        PREDICTION_COUNTER.labels("parent").inc()
        start = asyncio.get_event_loop().time()
        result, cached = await asyncio.to_thread(
            get_or_set_cache,
            "predict_parent",
            predict_parent,
            86400,
        )
        PREDICTION_LATENCY.labels("parent").observe(asyncio.get_event_loop().time() - start)
        return {"status": "completed", "cached": cached, "result": result}
    except PipelineError as e:
        if _not_found_for_missing_artifact(e):
            from src.pipelines.inference_pipeline import predict_parent
            from src.pipelines.train_pipeline import train_parent_model

            def chain_predict():
                get_or_set_cache("predict_parent", predict_parent, 86400)

            response.status_code = 202
            return await _start_auto_training(
                task_id="parent_training",
                train_fn=train_parent_model,
                chain_fn=chain_predict,
                detail="parent model missing. training started with auto-prediction",
            )
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.exception("Unexpected parent prediction failure")
        raise HTTPException(status_code=500, detail="parent prediction failed") from e


@router.post(
    "/predict-child",
    response_model=PredictionCompletedResponse | TrainingAcceptedResponse,
)
async def predict_child_endpoint(request: TickerRequest, response: Response):
    ticker = request.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")

    try:
        from src.pipelines.inference_pipeline import predict_child

        cache_key = f"predict_child:{ticker.lower()}"
        PREDICTION_COUNTER.labels("child").inc()
        start = asyncio.get_event_loop().time()
        result, cached = await asyncio.to_thread(
            get_or_set_cache,
            cache_key,
            lambda: predict_child(ticker),
            86400,
        )
        PREDICTION_LATENCY.labels("child").observe(asyncio.get_event_loop().time() - start)
        return {"status": "completed", "cached": cached, "result": result}
    except PipelineError as e:
        if _not_found_for_missing_artifact(e):
            from src.pipelines.inference_pipeline import predict_child
            from src.pipelines.train_pipeline import train_child_model

            task_id = ticker.lower()
            cache_key = f"predict_child:{task_id}"

            def chain_predict():
                get_or_set_cache(cache_key, lambda: predict_child(ticker), 86400)

            response.status_code = 202
            return await _start_auto_training_with_args(
                task_id=task_id,
                train_fn=train_child_model,
                train_args=(ticker,),
                chain_fn=chain_predict,
                detail=f"child model for {ticker} missing. training started with auto-prediction",
            )
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.exception(f"Unexpected child prediction failure for {ticker}")
        raise HTTPException(status_code=500, detail=f"child prediction failed for {ticker}") from e


@router.post(
    "/analyze",
    response_model=AnalyzeResponse | TrainingAcceptedResponse,
)
async def analyze(request: AnalyzeRequest, response: Response):
    ticker = request.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")

    prediction_cache_key = f"predict_child:{ticker.lower()}"
    ANALYSIS_COUNTER.inc()
    start = asyncio.get_event_loop().time()

    def build_analysis():
        from src.agents.langgraph_wrapper import analyze_stock

        return analyze_stock(
            ticker=ticker,
            thread_id=request.thread_id,
            use_fmi=request.use_fmi,
        )

    try:
        result = await asyncio.to_thread(build_analysis)
        duration = asyncio.get_event_loop().time() - start
        ANALYSIS_LATENCY.observe(duration)
        if isinstance(result, dict):
            metadata = result.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["duration_seconds"] = round(duration, 3)
            save_json(result, _latest_analysis_path(ticker))
        return result
    except PipelineError as e:
        if _not_found_for_missing_artifact(e):
            from src.pipelines.inference_pipeline import predict_child
            from src.pipelines.train_pipeline import train_child_model

            task_id = ticker.lower()

            def chain_predict():
                get_or_set_cache(prediction_cache_key, lambda: predict_child(ticker), 86400)

            response.status_code = 202
            return await _start_auto_training_with_args(
                task_id=task_id,
                train_fn=train_child_model,
                train_args=(ticker,),
                chain_fn=chain_predict,
                detail=f"analysis requested for {ticker}. child model missing, training started with auto-prediction",
            )
        ANALYSIS_FAILURES.inc()
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        ANALYSIS_FAILURES.inc()
        logger.exception(f"Unexpected analysis failure for {ticker}")
        raise HTTPException(status_code=500, detail=f"analysis failed for {ticker}") from e
