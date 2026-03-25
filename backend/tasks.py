import asyncio,time,json,logging
from datetime import datetime
from typing import Dict,Any,Optional
import psutil

import backend.state as app_state
from backend.state import (
    executor,
    TRAINING_DURATION,TRAINING_MSE,TRAINING_STATUS,SYSTEM_CPU,SYSTEM_DISK,
    SYSTEM_RAM,REDIS_KEYS,CACHE_HIT,CACHE_MISS
)

logger = logging.getLogger(__name__)

def _metric_key_label(key: str) -> str:
    return key if len(key) <= 100 else key[:97] + "..."


def get_or_set_cache(key,compute_fn,expire):
    refresh_system_metrics()
    try:
        if app_state.redis_client:
            val=app_state.redis_client.get(key)
            if val:
                CACHE_HIT.labels(_metric_key_label(key)).inc()
                return json.loads(val),True
        result=compute_fn()
        if app_state.redis_client:
            app_state.redis_client.set(key, json.dumps(result), ex=expire)
            CACHE_MISS.labels(_metric_key_label(key)).inc()
        return result, False
    except Exception as e:
        logger.error(f'redis cache error for key {key}: {e}')
        return compute_fn(), False
    
def refresh_system_metrics():
    SYSTEM_RAM.set(psutil.virtual_memory().used/(1024**2))
    SYSTEM_CPU.set(psutil.cpu_percent())
    SYSTEM_DISK.set(psutil.disk_usage('/').used/(1024**2))
    if app_state.redis_client:
        try:
            REDIS_KEYS.set(app_state.redis_client.dbsize())
        except:
            pass

def get_task_key(task_id: str) -> str:
    return f"task_status:{task_id.lower()}"

def save_task_status(task_id,status_data,ttl):
    try:
        if app_state.redis_client:
            app_state.redis_client.set(get_task_key(task_id),json.dumps(status_data),ex=ttl)
    except Exception as e:
        logging.error(f'failed to save task status for {task_id}')


def get_task_status_redis(task_id):
    try:
        if app_state.redis_client:
            val=app_state.redis_client.get(get_task_key(task_id))
            if val:
                return json.loads(val)
    except Exception as e:
        logging.error(f'failed to get task status for {task_id}')
    return None

async def run_training(task_id,fn,*args,chain_fn=None):
    curr_status=get_task_status_redis(task_id)
    if curr_status and curr_status.get('status')=='running':
        return 
    
    status_data={'status':'running','start_time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    save_task_status(task_id,status_data,ttl=3600)#1 hour max

    TRAINING_STATUS.labels(task_id).set(1)

    asyncio.create_task(run_training_worker(task_id,fn,*args,chain_fn=chain_fn))

async def run_training_worker(task_id,fn,*args,chain_fn):
    loop=asyncio.get_event_loop()
    start_time=time.time()
    try:
        result = await loop.run_in_executor(executor, fn, *args)

        if chain_fn:
            logger.info(f"Task {task_id}: Training complete, running chained task...")
            await loop.run_in_executor(executor, chain_fn)
            logger.info(f"Task {task_id}: Chained task complete.")

        duration = time.time() - start_time
        TRAINING_DURATION.labels(task_id).observe(duration)
        
        status_data = {"status": "completed", "result": result, "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        save_task_status(task_id, status_data, ttl=3600) # Keep completed status for 1 hour
        
        TRAINING_STATUS.labels(task_id).set(2)
        
        if isinstance(result, dict) and "mse" in result:
             TRAINING_MSE.set(result["mse"])
    except Exception as e:
        status_data = {"status": "failed", "error": str(e), "failed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        save_task_status(task_id, status_data, ttl=3600)
        
        TRAINING_STATUS.labels(task_id).set(0)
        logger.error(f"Training failed: {e}")

async def run_blocking_fn(fn,*args):
    loop=asyncio.get_event_loop()
    return await loop.run_in_executor(executor,fn,*args)
