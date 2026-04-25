import asyncio
import os
from contextlib import asynccontextmanager

import redis
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator

from backend.api_endpoints import router
from backend.rate_limit import check_rate_limit, rate_limit_response
from backend.state import REDIS_STATUS, registry
from backend.tasks import refresh_system_metrics
from logger.logger import get_logger
from src.utils import init_dir, setup_dagshub_mlflow
import backend.state as app_state

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_dir()
    setup_dagshub_mlflow()

    redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db = int(os.getenv("REDIS_DB", "0"))
    redis_connected = False

    for attempt in range(10):
        try:
            client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
            client.ping()
            app_state.redis_client = client
            REDIS_STATUS.set(1)
            redis_connected = True
            logger.info(
                "startup complete: Redis online at %s:%s (db=%s)",
                redis_host,
                redis_port,
                redis_db,
            )
            break
        except Exception:
            logger.warning(
                "waiting for redis at %s:%s... attempt %s/10",
                redis_host,
                redis_port,
                attempt + 1,
            )
            await asyncio.sleep(5)

    if not redis_connected:
        REDIS_STATUS.set(0)
        logger.error("startup continuing without Redis cache/status integration")

    try:
        yield
    finally:
        client = app_state.redis_client
        if client is not None:
            try:
                client.close()
            except Exception:
                logger.exception("failed to close Redis client cleanly")
        app_state.redis_client = None
        REDIS_STATUS.set(0)
        logger.info("application shutdown complete")


app = FastAPI(title="agentic-stock-pred-end2end", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def apply_rate_limit(request: Request, call_next):
    allowed, headers = check_rate_limit(request)
    if not allowed:
        return rate_limit_response(headers)

    response = await call_next(request)
    for key, value in headers.items():
        response.headers[key] = value
    return response


app.include_router(router)


@app.get("/metrics")
async def prometheus_metrics():
    refresh_system_metrics()
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


Instrumentator(registry=registry).instrument(app)

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
