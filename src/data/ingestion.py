from __future__ import annotations

import inspect
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from src.config import Config
from src.exception import PipelineError

load_dotenv()

YFINANCE_CACHE_DIR = Path.cwd() / ".cache" / "yfinance"
YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))

FEAST_VIEW_NAME = "stock_features"
FEAST_ENTITY_NAME = "ticker"
FEAST_FIELD_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "RSI": "rsi14",
    "MACD": "macd",
}


def macd(series: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _feature_repo_paths() -> tuple[Path, Path]:
    repo_path = Path.cwd() / "feature_store"
    data_path = repo_path / "data" / "features.parquet"
    return repo_path, data_path


def _ensure_feast_repo(repo_path: Path) -> None:
    repo_path.mkdir(parents=True, exist_ok=True)
    (repo_path / "data").mkdir(parents=True, exist_ok=True)

    config = Config()
    online_store = os.getenv("FEAST_ONLINE_STORE", config.feast_online_store).strip().lower()
    config_lines = [
        "project: stock_prediction",
        "provider: local",
        "registry: data/registry.db",
    ]

    if online_store == "redis":
        redis_host = os.getenv("REDIS_HOST", config.redis_host)
        redis_port = os.getenv("REDIS_PORT", str(config.redis_port))
        redis_db = os.getenv("REDIS_DB", str(config.redis_db))
        config_lines.extend(
            [
                "online_store:",
                "  type: redis",
                f"  connection_string: {redis_host}:{redis_port},db={redis_db}",
                "entity_key_serialization_version: 2",
            ]
        )

    (repo_path / "feature_store.yaml").write_text(
        "\n".join([*config_lines, ""]),
        encoding="utf-8",
    )


def _build_feast_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    feast_df = df[["date", *FEAST_FIELD_MAP.keys()]].copy()
    feast_df = feast_df.rename(columns=FEAST_FIELD_MAP)
    feast_df[FEAST_ENTITY_NAME] = ticker
    feast_df["event_timestamp"] = pd.to_datetime(feast_df.pop("date"))
    feast_df["created_timestamp"] = datetime.utcnow()

    return feast_df[
        [
            FEAST_ENTITY_NAME,
            "event_timestamp",
            "created_timestamp",
            *FEAST_FIELD_MAP.values(),
        ]
    ]


def _write_to_offline_store(feast_df: pd.DataFrame, data_path: Path) -> None:
    if data_path.exists():
        existing_df = pd.read_parquet(data_path)
        combined_df = pd.concat([existing_df, feast_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(
            subset=[FEAST_ENTITY_NAME, "event_timestamp"],
            keep="last",
        )
    else:
        combined_df = feast_df

    combined_df = combined_df.sort_values([FEAST_ENTITY_NAME, "event_timestamp"]).reset_index(drop=True)
    combined_df.to_parquet(data_path, index=False)


def _make_file_source(FileSource: type, data_path: Path):
    signature = inspect.signature(FileSource)
    kwargs = {
        "path": str(data_path),
        "created_timestamp_column": "created_timestamp",
    }

    if "timestamp_field" in signature.parameters:
        kwargs["timestamp_field"] = "event_timestamp"
    else:
        kwargs["event_timestamp_column"] = "event_timestamp"

    return FileSource(**kwargs)


def _register_and_materialize(repo_path: Path, data_path: Path) -> None:
    from feast import Entity, FeatureStore, FeatureView, Field, FileSource
    from feast.types import Float32, Int64

    entity_signature = inspect.signature(Entity)
    entity_kwargs = {"name": FEAST_ENTITY_NAME}
    if "value_type" in entity_signature.parameters:
        from feast import ValueType

        entity_kwargs["value_type"] = ValueType.STRING

    if "join_keys" in entity_signature.parameters:
        stock_entity = Entity(join_keys=[FEAST_ENTITY_NAME], **entity_kwargs)
    else:
        stock_entity = Entity(join_key=FEAST_ENTITY_NAME, **entity_kwargs)

    stock_source = _make_file_source(FileSource, data_path)
    stock_features = FeatureView(
        name=FEAST_VIEW_NAME,
        entities=[stock_entity],
        ttl=timedelta(days=365),
        schema=[
            Field(name="open", dtype=Float32),
            Field(name="high", dtype=Float32),
            Field(name="low", dtype=Float32),
            Field(name="close", dtype=Float32),
            Field(name="volume", dtype=Int64),
            Field(name="rsi14", dtype=Float32),
            Field(name="macd", dtype=Float32),
        ],
        source=stock_source,
    )

    store = FeatureStore(repo_path=str(repo_path))
    store.apply([stock_entity, stock_features])
    store.materialize_incremental(end_date=datetime.utcnow())


def _validate_data(df: pd.DataFrame, ticker: str, config: Config) -> pd.DataFrame:
    expected_columns = ["date", *config.features]
    missing_columns = [column for column in expected_columns if column not in df.columns]
    if missing_columns:
        raise PipelineError(f"missing columns for {ticker}: {missing_columns}")

    df = df[expected_columns].dropna().copy()
    if len(df) < config.context_len + config.pred_len:
        raise PipelineError(f"not enough data to create sequences for {ticker}")

    if df[config.features].isna().any().any():
        raise PipelineError(f"missing values in features for {ticker}")

    non_numeric = [column for column in config.features if not pd.api.types.is_numeric_dtype(df[column])]
    if non_numeric:
        raise PipelineError(f"non-numeric values found in features for {ticker}: {non_numeric}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise PipelineError(f"date column is not in datetime format for {ticker}")

    if not df["date"].is_monotonic_increasing:
        raise PipelineError(f"date column is not in ascending order for {ticker}")

    return df.reset_index(drop=True)


def fetch_ohlcv(ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    config = Config()
    start_date = start or config.start_date

    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            raise PipelineError(f"no data found for {ticker}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index().rename(columns={"Date": "date"})
        df["RSI"] = rsi(df["Close"])
        df["MACD"] = macd(df["Close"])
        df = _validate_data(df, ticker, config)

        repo_path, data_path = _feature_repo_paths()
        _ensure_feast_repo(repo_path)

        feast_df = _build_feast_df(df, ticker)
        _write_to_offline_store(feast_df, data_path)
        _register_and_materialize(repo_path, data_path)

        return df
    except Exception as e:
        raise PipelineError(f"failed to fetch ohlcv data for {ticker}: {e}") from e
