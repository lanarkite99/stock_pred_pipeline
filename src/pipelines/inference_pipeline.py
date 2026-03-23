import joblib
import logging
import os
import pickle

import torch
from feast import FeatureStore

from src.config import Config
from src.data.ingestion import fetch_ohlcv
from src.exception import PipelineError
from src.inference import predict_one_step_and_week
from src.model.definition import StockLSTM

logger = logging.getLogger(__name__)
config = Config()

ONLINE_FEATURES = [
    "stock_features:open",
    "stock_features:high",
    "stock_features:low",
    "stock_features:close",
    "stock_features:volume",
    "stock_features:rsi14",
    "stock_features:macd",
]


def get_feature_store():
    try:
        return FeatureStore(repo_path="feature_store")
    except Exception as e:
        logger.warning(f"Feast feature store not initialized: {e}")
        return None


def _safe_load_scaler(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        try:
            return joblib.load(path)
        except Exception as e:
            raise PipelineError(f"Scaler load failed ({path}): {e}") from e


def _load_local_model(ticker: str, model_type: str):
    try:
        if model_type == "parent":
            base_dir = config.parent_dir
            model_path = os.path.join(base_dir, f"{config.parent_ticker}_parent_model.pt")
            scaler_path = os.path.join(base_dir, f"{config.parent_ticker}_parent_scaler.pkl")
        else:
            base_dir = os.path.join(config.workdir, ticker.lower())
            model_path = os.path.join(base_dir, f"{ticker}_child_model.pt")
            scaler_path = os.path.join(base_dir, f"{ticker}_child_scaler.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"missing PyTorch model for {ticker}: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"missing scaler for {ticker}: {scaler_path}")

        model = StockLSTM(input_size=config.input_size, pred_len=config.pred_len).to(config.device)
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        model.eval()

        scaler = _safe_load_scaler(scaler_path)
        logger.info(f"Loaded {model_type} model for {ticker}")
        return model, scaler
    except Exception as e:
        raise PipelineError(f"Local model load failed for {ticker}: {e}") from e


def _attach_history(preds: dict, df):
    history_df = df.tail(30).copy()
    history_df.columns = [column.lower() for column in history_df.columns]

    if "date" in history_df.columns:
        history_df["date"] = history_df["date"].astype(str)
        preds["history"] = history_df[["date", "close"]].to_dict(orient="records")
    else:
        preds["history"] = [
            {"date": str(index.date()), "close": row["close"]}
            for index, row in history_df.iterrows()
        ]
    return preds


def _fetch_online_features(ticker: str) -> None:
    store = get_feature_store()
    if not store:
        return

    try:
        feature_vector = store.get_online_features(
            features=ONLINE_FEATURES,
            entity_rows=[{"ticker": ticker}],
        ).to_dict()
        logger.info(f"Fetched online features from Feast for {ticker}: {feature_vector}")
    except Exception as e:
        logger.warning(f"Failed to fetch online features for {ticker}: {e}")


def predict_parent():
    try:
        ticker = config.parent_ticker
        model, scaler = _load_local_model(ticker, "parent")
        df = fetch_ohlcv(ticker)
        _fetch_online_features(ticker)

        preds = predict_one_step_and_week(model, df, scaler, ticker)
        preds = _attach_history(preds, df)
        logger.info(f"Parent prediction completed for {ticker}")
        return preds
    except Exception as e:
        logger.error(f"Parent prediction failed: {e}")
        raise PipelineError(f"Parent prediction failed: {e}") from e


def predict_child(ticker: str):
    try:
        model, scaler = _load_local_model(ticker, "child")
        df = fetch_ohlcv(ticker)
        _fetch_online_features(ticker)

        preds = predict_one_step_and_week(model, df, scaler, ticker)
        preds = _attach_history(preds, df)
        logger.info(f"Child prediction completed for {ticker}")
        return preds
    except Exception as e:
        logger.error(f"Child prediction failed for {ticker}: {e}")
        raise PipelineError(f"Child prediction failed for {ticker}: {e}") from e
