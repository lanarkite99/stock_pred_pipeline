import asyncio
import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from backend.api_endpoints import router
from src.utils import init_dir, setup_dagshub_mlflow

logger = logging.getLogger(__name__)

try:
    import redis
except ImportError:  # pragma: no cover - optional until Redis layer is added
    redis = None

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
except ImportError:  # pragma: no cover - optional until metrics layer is added
    Instrumentator = None
    CONTENT_TYPE_LATEST = "text/plain; charset=utf-8"
    generate_latest = None

try:
    from backend.state import REDIS_STATUS, registry
    import backend.state as app_state
except ImportError:  # pragma: no cover - optional until backend state module is added
    REDIS_STATUS = None
    registry = None
    app_state = None

try:
    from backend.tasks import refresh_system_metrics
except ImportError:  # pragma: no cover - optional until background tasks module is added
    refresh_system_metrics = None


def _try_setup_mlflow() -> None:
    try:
        setup_dagshub_mlflow()
    except Exception as exc:
        logger.warning(f"MLflow/DagsHub setup skipped: {exc}")


async def _try_connect_redis() -> None:
    if redis is None or app_state is None:
        logger.info("Redis integration not enabled yet; skipping startup connection.")
        return

    for attempt in range(10):
        try:
            client = redis.Redis(host="redis", port=6379, db=0)
            client.ping()
            app_state.redis_client = client
            if REDIS_STATUS is not None:
                REDIS_STATUS.set(1)
            logger.info("Redis connected successfully.")
            return
        except Exception:
            logger.warning(f"Waiting for Redis... attempt {attempt + 1}/10")
            await asyncio.sleep(5)

    if REDIS_STATUS is not None:
        REDIS_STATUS.set(0)
    logger.warning("Redis not available. API will continue without Redis-dependent features.")


app = FastAPI(title="agentic-stock-pred-end2end")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if Instrumentator is not None and registry is not None:
    Instrumentator(registry=registry).instrument(app).expose(app, include_in_schema=False)


@app.on_event("startup")
async def startup() -> None:
    init_dir()
    _try_setup_mlflow()
    await _try_connect_redis()


@app.get("/metrics")
async def prometheus_metrics():
    if refresh_system_metrics is not None:
        try:
            refresh_system_metrics()
        except Exception as exc:
            logger.warning(f"Metrics refresh failed: {exc}")

    if generate_latest is None or registry is None:
        return Response("metrics not configured yet", media_type="text/plain")

    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
