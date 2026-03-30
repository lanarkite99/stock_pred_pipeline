from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from logger.logger import get_logger

logger = get_logger()


def _fetch_ohlcv(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    try:
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        cols = [col for col in ["Open", "High", "Low", "Close", "Volume"] if col in df.columns]
        return df[cols].dropna()
    except Exception as e:
        logger.warning("regime data fetch failed for %s: %s", ticker, e)
        return pd.DataFrame()


def _score_shift(reference: pd.Series, current: pd.Series) -> float:
    ref_mean = float(reference.mean())
    ref_std = float(reference.std())
    curr_mean = float(current.mean())
    return abs(curr_mean - ref_mean) / (ref_std + 1e-9)


def assess_regime(ticker: str) -> dict[str, Any]:
    now = datetime.now()
    current_start = now - timedelta(days=30)
    reference_start = now - timedelta(days=180)

    ref_df = _fetch_ohlcv(ticker, reference_start, current_start)
    curr_df = _fetch_ohlcv(ticker, current_start, now + timedelta(days=1))

    if len(ref_df) < 20 or len(curr_df) < 5:
        return {
            "status": "warning",
            "reference_window": {"start": reference_start.strftime("%Y-%m-%d"), "end": current_start.strftime("%Y-%m-%d")},
            "current_window": {"start": current_start.strftime("%Y-%m-%d"), "end": now.strftime("%Y-%m-%d")},
            "metrics": {},
            "notes": ["Insufficient market data for regime assessment."],
        }

    ref_returns = ref_df["Close"].pct_change().dropna()
    curr_returns = curr_df["Close"].pct_change().dropna()

    return_drift_score = _score_shift(ref_returns, curr_returns) if not ref_returns.empty and not curr_returns.empty else 0.0
    volatility_ratio = float(curr_returns.std() / (ref_returns.std() + 1e-9)) if not ref_returns.empty and not curr_returns.empty else 1.0
    volume_shift_score = _score_shift(ref_df["Volume"], curr_df["Volume"])
    ref_range = (ref_df["High"] - ref_df["Low"]) / ref_df["Close"].replace(0, np.nan)
    curr_range = (curr_df["High"] - curr_df["Low"]) / curr_df["Close"].replace(0, np.nan)
    price_range_shift_score = _score_shift(ref_range.dropna(), curr_range.dropna()) if not ref_range.dropna().empty and not curr_range.dropna().empty else 0.0

    notes: list[str] = []
    if volatility_ratio > 1.5 or volatility_ratio < 0.67:
        notes.append("Volatility regime has shifted versus the reference window.")
    if return_drift_score > 1.0:
        notes.append("Recent return behavior differs meaningfully from the reference period.")
    if volume_shift_score > 1.0:
        notes.append("Recent trading volume is outside the reference regime.")

    status = "healthy"
    if return_drift_score > 2.0 or volatility_ratio > 2.0 or volatility_ratio < 0.5:
        status = "critical"
    elif return_drift_score > 1.0 or volatility_ratio > 1.5 or volatility_ratio < 0.67 or volume_shift_score > 1.0:
        status = "warning"

    if not notes:
        notes.append("Recent market behavior remains broadly aligned with the reference window.")

    return {
        "status": status,
        "reference_window": {"start": reference_start.strftime("%Y-%m-%d"), "end": current_start.strftime("%Y-%m-%d")},
        "current_window": {"start": current_start.strftime("%Y-%m-%d"), "end": now.strftime("%Y-%m-%d")},
        "metrics": {
            "return_drift_score": round(return_drift_score, 4),
            "volatility_ratio": round(volatility_ratio, 4),
            "volume_shift_score": round(volume_shift_score, 4),
            "price_range_shift_score": round(price_range_shift_score, 4),
        },
        "notes": notes,
    }
