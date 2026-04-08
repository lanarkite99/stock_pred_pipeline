import logging
from typing import Dict

import pandas as pd
import torch
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.exception import PipelineError

logger = logging.getLogger(__name__)

NSE_HOLIDAYS = {
    "2025-02-26",
    "2025-03-14",
    "2025-03-31",
    "2025-04-10",
    "2025-04-14",
    "2025-04-18",
    "2025-05-01",
    "2025-08-15",
    "2025-08-27",
    "2025-10-02",
    "2025-10-21",
    "2025-10-22",
    "2025-11-05",
    "2025-12-25",
    "2026-02-15",
    "2026-02-19",
    "2026-03-04",
    "2026-03-26",
    "2026-04-02",
    "2026-04-03",
    "2026-04-14",
    "2026-05-01",
    "2026-09-17",
    "2026-10-02",
    "2026-10-20",
    "2026-11-09",
    "2026-11-16",
    "2026-12-25",
}


def _next_trading_days(last_date: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    nse_business_day = CustomBusinessDay(holidays=sorted(NSE_HOLIDAYS))
    return pd.date_range(last_date + nse_business_day, periods=periods, freq=nse_business_day)


def _normalize_forecast_row(row: dict) -> tuple[dict, list[str]]:
    normalized = dict(row)
    warnings: list[str] = []

    open_price = float(normalized["open"])
    high_price = float(normalized["high"])
    low_price = float(normalized["low"])
    close_price = float(normalized["close"])
    volume = max(0.0, float(normalized["volume"]))

    if high_price < low_price:
        high_price, low_price = low_price, high_price
        warnings.append("high_below_low_swapped")

    max_price = max(open_price, high_price, low_price, close_price)
    min_price = min(open_price, high_price, low_price, close_price)

    if high_price < max_price:
        high_price = max_price
        warnings.append("high_adjusted")

    if low_price > min_price:
        low_price = min_price
        warnings.append("low_adjusted")

    if open_price < low_price:
        open_price = low_price
        warnings.append("open_clamped")
    elif open_price > high_price:
        open_price = high_price
        warnings.append("open_clamped")

    if close_price < low_price:
        close_price = low_price
        warnings.append("close_clamped")
    elif close_price > high_price:
        close_price = high_price
        warnings.append("close_clamped")

    normalized.update(
        {
            "open": float(open_price),
            "high": float(high_price),
            "low": float(low_price),
            "close": float(close_price),
            "volume": float(volume),
        }
    )
    return normalized, warnings


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
        next_days = _next_trading_days(last_date, config.pred_len)

        forecast = []
        sanity_adjustments = []
        for i, date in enumerate(next_days):
            raw_row = {
                "date": str(date.date()),
                "open": float(preds_inv[i][0]),
                "high": float(preds_inv[i][1]),
                "low": float(preds_inv[i][2]),
                "close": float(preds_inv[i][3]),
                "volume": float(preds_inv[i][4]),
            }
            normalized_row, warnings = _normalize_forecast_row(raw_row)
            if warnings:
                sanity_adjustments.append(
                    {
                        "date": normalized_row["date"],
                        "warnings": warnings,
                        "raw": raw_row,
                        "normalized": normalized_row,
                    }
                )
            forecast.append(normalized_row)

        if sanity_adjustments:
            logger.warning(
                "Applied forecast sanity checks for %s: %s",
                ticker,
                ", ".join(item["date"] for item in sanity_adjustments),
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
                "sanity_checks": {
                    "adjusted_rows": len(sanity_adjustments),
                    "adjustments": sanity_adjustments,
                },
            },
        }
    except Exception as e:
        logger.error(f"Prediction failed for {ticker}: {e}")
        raise PipelineError(f"Prediction failed for {ticker}: {e}") from e

