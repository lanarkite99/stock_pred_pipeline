from typing import Any

from pydantic import BaseModel, Field


class TickerRequest(BaseModel):
    ticker: str = Field(..., min_length=1, description="Yahoo Finance ticker, e.g. RELIANCE.NS")


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., min_length=1, description="Yahoo Finance ticker to analyze")
    use_fmi: bool = False
    thread_id: str | None = None


class HealthResponse(BaseModel):
    status: str


class TrainingAcceptedResponse(BaseModel):
    status: str
    task_id: str
    detail: str | None = None


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None
    start_time: str | None = None
    completed_at: str | None = None
    failed_at: str | None = None


class PredictionCompletedResponse(BaseModel):
    status: str
    cached: bool
    result: dict[str, Any]


class RootResponse(BaseModel):
    project: str
    description: str
    features: list[str]
    endpoints: dict[str, str]
    quick_start: dict[str, dict[str, Any]]


class AnalyzeMetadata(BaseModel):
    source_cache: str
    llm_model: str
    embed_model: str | None = None
    duration_seconds: float | None = None
    ticker: str
    news_empty: bool


class AnalyzeResponse(BaseModel):
    status: str
    ticker: str
    recommendation: str
    confidence: str
    summary: str
    final_report: str | None = None
    news_sentiment: str | None = None
    prediction: dict[str, Any]
    thread_id: str | None = None
    use_fmi: bool = False
    cached: bool = False
    metadata: AnalyzeMetadata | None = None


class MonitorResponse(BaseModel):
    ticker: str
    status: str
    timestamp: str
    summary: dict[str, str]
    system: dict[str, Any]
    regime: dict[str, Any]
    analysis_quality: dict[str, Any]
    artifacts: dict[str, str]
