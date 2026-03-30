import os
from typing import Any

import requests
import redis

from logger.logger import get_logger
from src.memory.semantic_cache import SemanticCache

logger = get_logger()


def _redis_client() -> redis.Redis | None:
    host = os.getenv("REDIS_HOST", "127.0.0.1")
    port = int(os.getenv("REDIS_PORT", "6379"))
    db = int(os.getenv("REDIS_DB", "0"))
    try:
        client = redis.Redis(host=host, port=port, db=db)
        client.ping()
        return client
    except Exception:
        return None


def _redis_up() -> bool:
    return _redis_client() is not None


def _ollama_reachable() -> bool:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def _prediction_cache_available(ticker: str) -> bool:
    client = _redis_client()
    if client is None:
        return False
    try:
        return bool(client.exists(f"predict_child:{ticker.lower()}"))
    except Exception:
        return False
    finally:
        try:
            client.close()
        except Exception:
            pass


def _semantic_cache_available() -> bool:
    try:
        cache = SemanticCache(collection_name="analysis_cache")
        cache.collection.count()
        return True
    except Exception as e:
        logger.warning("semantic cache availability check failed: %s", e)
        return False


def check_system_health(ticker: str) -> dict[str, Any]:
    result = {
        "api_healthy": True,
        "redis_up": _redis_up(),
        "ollama_reachable": _ollama_reachable(),
        "prediction_cache_available": _prediction_cache_available(ticker),
        "semantic_cache_available": _semantic_cache_available(),
    }
    failures = sum(1 for value in result.values() if not value)
    result["status"] = "healthy" if failures == 0 else "warning" if failures <= 2 else "critical"
    return result
