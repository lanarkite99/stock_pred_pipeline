import os,json,base64,mlflow,logging
import pandas as pd
from io import BytesIO
from typing import Dict
from src.config import Config
from src.exception import PipelineError
from dotenv import load_dotenv

logger=logging.getLogger(__name__)

def setup_dagshub_mlflow():
    load_dotenv()
    dagshub_user=os.getenv("DAGSHUB_USER_NAME")
    dagshub_repo=os.getenv("DAGSHUB_REPO_NAME")
    dagshub_token=os.getenv("DAGSHUB_TOKEN")
    mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI")

    if dagshub_user and dagshub_repo:
        try:
            import dagshub
            import dagshub.auth
            
            # Authenticate if token is present
            if dagshub_token:
                try:
                    dagshub.auth.add_app_token(dagshub_token)
                    logger.info("DagsHub token added")
                except Exception as e:
                    if "File exists" in str(e):
                        logger.info("DagsHub token already exists")
                    else:
                        logger.warning(f"failed to add DagsHub token: {e}")

            #Init DagsHub
            dagshub.init(repo_owner=dagshub_user, repo_name=dagshub_repo, mlflow=True)
            
            # Set MLflow tracking URI from .env
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                logger.info(f"DagsHub MLflow tracking initialized: {mlflow_tracking_uri}")
            else:
                # Fallback to constructed URI
                dagshub_mlflow_uri = f"https://dagshub.com/{dagshub_user}/{dagshub_repo}.mlflow"
                mlflow.set_tracking_uri(dagshub_mlflow_uri)
                logger.info(f"DagsHub MLflow tracking initialized: {dagshub_mlflow_uri}")
            try:
                registry_uri = mlflow.get_tracking_uri()
                mlflow.set_registry_uri(registry_uri)
                logger.info(f"MLflow model registry initialized: {registry_uri}")
            except Exception as e:
                logger.warning(f"Failed setting MLflow registry URI: {e}")
            
            # Set authentication credentials for MLflow
            if dagshub_token:
                os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_user
                os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
                logger.info("DagsHub auth configured")
            else:
                logger.warning("DAGSHUB_TOKEN not set, you may have read only access")
            
            return True
        except ImportError:
                logger.warning("dagshub package not installed. Install with: pip install dagshub/uv add dagshub")
                logger.info("Please install: pip install dagshub/uv add dagshub")
        except Exception as e:
            logger.warning(f"Failed to initialize DagsHub: {e}")
    else:
        logger.warning("DAGSHUB_USER_NAME or DAGSHUB_REPO_NAME not set in .env file")
    
    return False

def save_json(data:dict, path:str):
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,'w') as f:
        json.dump(data,f,indent=1)
    return path

def init_dir():
    config=Config()
    os.makedirs(config.parent_dir, exist_ok=True)

            
