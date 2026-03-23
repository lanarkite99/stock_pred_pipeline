import logging
from typing import Dict

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.exception import PipelineError

logger = logging.getLogger(__name__)


def predict_one_step_and_week(model, df: pd.DataFrame, scaler: StandardScaler, ticker: str) -> Dict:
    try:
        config = Config()
        if len(df) < config.context_len:
            raise PipelineError(f"not enough rows to predict for {ticker}")

        feature_values = scaler.transform(df[config.features].values).astype("float32")
        model_input = feature_values[-config.context_len:].reshape(1, config.context_len, config.input_size)

        with torch.no_grad():
            input_tensor = torch.tensor(model_input, dtype=torch.float32).to(config.device)
            preds = model(input_tensor).cpu().numpy()[0]

        preds_inv = scaler.inverse_transform(preds.reshape(-1, config.input_size))

        last_date = pd.to_datetime(df["date"].iloc[-1])
        next_days = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=config.pred_len)

        forecast = []
        for i, date in enumerate(next_days):
            forecast.append(
                {
                    "date": str(date.date()),
                    "open": float(preds_inv[i][0]),
                    "high": float(preds_inv[i][1]),
                    "low": float(preds_inv[i][2]),
                    "close": float(preds_inv[i][3]),
                    "volume": float(preds_inv[i][4]),
                }
            )

        return {
            "ticker": ticker,
            "last_date": str(last_date.date()),
            "future_window_days": config.pred_len,
            "next_business_days": [str(day.date()) for day in next_days],
            "predictions": {
                "next_day": forecast[0],
                "next_week": {
                    "high": float(max(item["high"] for item in forecast)),
                    "low": float(min(item["low"] for item in forecast)),
                },
                "full_forecast": forecast,
            },
        }
    except Exception as e:
        logger.error(f"Prediction failed for {ticker}: {e}")
        raise PipelineError(f"Prediction failed for {ticker}: {e}") from e
