from typing import Any

from logger.logger import get_logger
from src.agents.langgraph_wrapper import analyze_stock

logger = get_logger()


def _stance_consistent(result: dict[str, Any]) -> bool:
    prediction = result.get("prediction", {})
    history = prediction.get("history", [])
    next_day = prediction.get("predictions", {}).get("next_day", {})
    if not history or not next_day:
        return False

    last_close = float(history[-1].get("close", 0.0))
    next_close = float(next_day.get("close", 0.0))
    recommendation = str(result.get("recommendation", "NEUTRAL")).upper()

    if next_close > last_close * 1.002:
        return recommendation in {"BULLISH", "NEUTRAL"}
    if next_close < last_close * 0.998:
        return recommendation in {"BEARISH", "NEUTRAL"}
    return recommendation == "NEUTRAL"


def _forecast_grounded(result: dict[str, Any]) -> bool:
    report = str(result.get("final_report", ""))
    prediction = result.get("prediction", {})
    next_week = prediction.get("predictions", {}).get("next_week", {})
    values = [next_week.get("low"), next_week.get("high")]
    numeric_tokens = {f"{float(v):.2f}" for v in values if v is not None}
    return any(token in report for token in numeric_tokens)


def evaluate_analysis_quality(ticker: str) -> dict[str, Any]:
    try:
        result = analyze_stock(ticker)
    except Exception as e:
        logger.warning("analysis quality check failed for %s: %s", ticker, e)
        return {
            "status": "critical",
            "score": 0.0,
            "checks": {"analysis_ran": False},
            "metadata": {"ticker": ticker.upper()},
        }

    metadata = result.get("metadata", {})
    report = str(result.get("final_report", ""))
    recommendation = str(result.get("recommendation", "")).upper()
    confidence = str(result.get("confidence", ""))
    news_empty = bool(metadata.get("news_empty", False))

    checks = {
        "has_recommendation": recommendation in {"BULLISH", "BEARISH", "NEUTRAL"},
        "has_confidence": confidence in {"High", "Medium", "Low"},
        "forecast_grounded": _forecast_grounded(result),
        "stance_consistent": _stance_consistent(result),
        "news_honest": ("verify current sector" in report.lower() or "news signal is limited" in report.lower()) if news_empty else bool(result.get("news_sentiment")),
        "report_nonempty": len(report.strip()) > 120,
    }

    score = round(sum(checks.values()) / len(checks), 2)
    status = "healthy" if score >= 0.8 else "warning" if score >= 0.5 else "critical"

    return {
        "status": status,
        "score": score,
        "checks": checks,
        "metadata": {
            "source_cache": metadata.get("source_cache", "unknown"),
            "llm_model": metadata.get("llm_model", ""),
            "embed_model": metadata.get("embed_model"),
            "duration_seconds": metadata.get("duration_seconds"),
            "ticker": metadata.get("ticker", ticker.upper()),
            "news_empty": news_empty,
        },
    }
