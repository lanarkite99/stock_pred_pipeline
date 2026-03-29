from concurrent.futures import ThreadPoolExecutor
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram
import redis
from typing import Dict,Any, Optional

redis_client:Optional[redis.Redis]=None
executor=ThreadPoolExecutor(max_workers=4)

registry = CollectorRegistry()
SYSTEM_CPU = Gauge("system_cpu_percent", "CPU percent", registry=registry)
SYSTEM_RAM = Gauge("system_ram_used_mb", "RAM MB", registry=registry)
SYSTEM_DISK = Gauge("system_disk_used_mb", "Disk Used MB", registry=registry)
REDIS_STATUS = Gauge("redis_up", "Redis up=1/down=0", registry=registry)
REDIS_KEYS = Gauge("redis_keys_total", "Number of keys in Redis", registry=registry)
TRAINING_STATUS = Gauge("training_status", "0=idle 1=running 2=completed", ["task_id"], registry=registry)
TRAINING_MSE = Gauge("training_mse_last", "Last training MSE", registry=registry)
TRAINING_DURATION = Histogram("training_duration_seconds", "Training duration in seconds", ["task_id"], registry=registry)
PREDICTION_COUNTER = Counter("prediction_total", "Total predictions", ["type"], registry=registry)
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency", ["type"], registry=registry)
CACHE_HIT = Counter("redis_cache_hit_total", "Cache hits", ["key"], registry=registry)
CACHE_MISS = Counter("redis_cache_miss_total", "Cache misses", ["key"], registry=registry)
ANALYSIS_COUNTER = Counter("analysis_total", "Total analysis requests", registry=registry)
ANALYSIS_FAILURES = Counter("analysis_failures_total", "Total failed analysis requests", registry=registry)
ANALYSIS_LATENCY = Histogram("analysis_latency_seconds", "Analysis latency in seconds", registry=registry)
ANALYSIS_CACHE_HIT = Counter("analysis_cache_hit_total", "Semantic analysis cache hits", registry=registry)
ANALYSIS_CACHE_MISS = Counter("analysis_cache_miss_total", "Semantic analysis cache misses", registry=registry)
