import asyncio,os,redis, uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from backend.state import registry, REDIS_STATUS
from backend.tasks import refresh_system_metrics
import backend.state as app_state
from backend.api_endpoints import router

from src.utils import setup_dagshub_mlflow,init_dir
import logging

logger = logging.getLogger(__name__)

setup_dagshub_mlflow()
app=FastAPI(title='agentic-stock-pred-end2end')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

app.include_router(router)

@app.get('/metrics')
async def prometheus_metrics():
    refresh_system_metrics()
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

Instrumentator(registry=registry).instrument(app)

@app.on_event('startup')
async def startup():
    init_dir()
    redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db = int(os.getenv("REDIS_DB", "0"))
    for i in range(10):
        try:
            client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
            client.ping()
            app_state.redis_client = client
            REDIS_STATUS.set(1)
            logger.info(f"systems online (Redis at {redis_host}:{redis_port}, MLflow)")
            return
        except Exception as e:
            logger.warning(f"waiting for redis at {redis_host}:{redis_port}... attempt {i+1}/10")
            await asyncio.sleep(5)
            
    REDIS_STATUS.set(0)
    logger.error("failed to connect to Redis after multiple attempts")

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
