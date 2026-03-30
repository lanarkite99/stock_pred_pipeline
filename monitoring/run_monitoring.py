from datetime import datetime
from typing import Any

from logger.logger import get_logger
from monitoring.health_checks import check_system_health
from monitoring.regime import assess_regime
from monitoring.response_quality import evaluate_analysis_quality
from src.config import Config
from src.utils import save_json

logger = get_logger()


def _overall_status(system: dict[str, Any], regime: dict[str, Any], quality: dict[str, Any]) -> str:
    statuses = [system.get("status", "healthy"), regime.get("status", "healthy"), quality.get("status", "healthy")]
    if "critical" in statuses:
        return "critical"
    if "warning" in statuses:
        return "warning"
    return "healthy"


def run_monitoring(ticker: str) -> dict[str, Any]:
    ticker_u = ticker.upper()
    config = Config()

    system = check_system_health(ticker_u)
    regime = assess_regime(ticker_u)
    analysis_quality = evaluate_analysis_quality(ticker_u)
    status = _overall_status(system, regime, analysis_quality)

    result = {
        "ticker": ticker_u,
        "status": status,
        "timestamp": datetime.now().astimezone().isoformat(),
        "summary": {
            "system": system.get("status", "healthy"),
            "regime": regime.get("status", "healthy"),
            "analysis_quality": analysis_quality.get("status", "healthy"),
        },
        "system": system,
        "regime": regime,
        "analysis_quality": analysis_quality,
        "artifacts": {
            "analysis_output": f"{config.workdir}/{ticker.lower()}/latest_analysis.json",
            "training_summary": f"{config.workdir}/{ticker.lower()}/{ticker_u}_child_training_summary.json",
        },
    }

    output_path = f"{config.workdir}/{ticker.lower()}/monitor/latest_monitor.json"
    save_json(result, output_path)
    logger.info("monitoring summary saved for %s -> %s", ticker_u, output_path)
    return result
