import json
import os

import boto3
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph

from src.agents.agents import perf_analyst, report_generator
from src.agents.fetch import fetch_pred_data
from src.exception import PipelineError
from src.memory.semantic_cache import SemanticCache
from backend.state import ANALYSIS_CACHE_HIT, ANALYSIS_CACHE_MISS
from logger.logger import get_logger

logger = get_logger()


class AgentState(MessagesState):
    ticker: str
    preds: str
    perf_summary: str
    recommendation: str
    confidence: str
    news_sentiment: str
    final_report: str


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("perf", perf_analyst)
    g.add_node("report", report_generator)

    g.set_entry_point("perf")
    g.add_edge("perf", "report")
    g.add_edge("report", END)

    return g.compile(checkpointer=MemorySaver())


class BedrockEmbedder:
    def __init__(self):
        self.model_id = os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v1")
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "ap-south-1"),
        )

    def embed_query(self, text: str) -> list[float]:
        response = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": text}),
        )
        payload = json.loads(response["body"].read())
        return payload.get("embedding", [])


def _get_embedder():
    try:
        return BedrockEmbedder()
    except Exception as e:
        logger.warning("semantic cache embedder init failed: %s", e)
        return None


def _llm_model_name() -> str:
    return os.getenv("BEDROCK_CHAT_MODEL_ID", "openai.gpt-oss-20b-1:0")


def _embed_model_name(embedder) -> str | None:
    if embedder is None:
        return None
    return getattr(embedder, "model_id", os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v1"))


def _news_is_empty(news_sentiment: str) -> bool:
    normalized = (news_sentiment or "").strip().lower()
    return not normalized or normalized.startswith("no recent news")


def _should_refresh_cached_result(ticker: str, cached_row: dict) -> bool:
    news_sentiment = cached_row.get("news_sentiment", "")
    cached_news_empty = str(cached_row.get("news_empty", "")).lower() == "true" or _news_is_empty(news_sentiment)
    return ticker.upper().endswith(".NS") and cached_news_empty


def _build_metadata(
    *,
    ticker: str,
    source_cache: str,
    news_sentiment: str,
    embedder,
    llm_model: str | None = None,
    embed_model: str | None = None,
) -> dict:
    return {
        "source_cache": source_cache,
        "llm_model": llm_model or _llm_model_name(),
        "embed_model": embed_model if embed_model is not None else _embed_model_name(embedder),
        "ticker": ticker,
        "news_empty": _news_is_empty(news_sentiment),
    }


def analyze_stock(ticker, thread_id=None, use_fmi=False):
    ticker_u = ticker.upper()
    embedder = _get_embedder()
    query_vec = None

    if embedder is not None:
        try:
            cache = SemanticCache(collection_name="analysis_cache")
            query_text = f"Analysis report for {ticker_u}"
            query_vec = embedder.embed_query(query_text)
            hits = cache.recall(query_vec, ticker=ticker_u, limit=3)
            if hits:
                hits.sort(key=lambda item: int(item.get("created_at_ts", 0)), reverse=True)
                best = hits[0]
                if _should_refresh_cached_result(ticker_u, best):
                    ANALYSIS_CACHE_MISS.inc()
                    logger.info("semantic cache stale for %s due to empty news; rebuilding analysis", ticker_u)
                else:
                    ANALYSIS_CACHE_HIT.inc()
                    logger.info("semantic cache hit for %s", ticker_u)
                    return {
                        "status": "completed",
                        "ticker": ticker_u,
                        "recommendation": best.get("recommendation", "NEUTRAL"),
                        "confidence": best.get("confidence", "Medium"),
                        "summary": best.get("summary", ""),
                        "final_report": best.get("final_report", best.get("summary", "")),
                        "news_sentiment": best.get("news_sentiment", ""),
                        "prediction": __import__("json").loads(best.get("prediction_json", "{}")),
                        "thread_id": best.get("thread_id") or thread_id,
                        "use_fmi": bool(best.get("use_fmi", use_fmi)),
                        "cached": True,
                        "metadata": _build_metadata(
                            ticker=ticker_u,
                            source_cache="semantic_cache",
                            news_sentiment=best.get("news_sentiment", ""),
                            embedder=embedder,
                            llm_model=best.get("llm_model") or None,
                            embed_model=best.get("embed_model") or None,
                        ),
                    }
            ANALYSIS_CACHE_MISS.inc()
            logger.info("semantic cache miss for %s", ticker_u)
        except Exception as e:
            logger.exception("semantic cache recall failed for %s: %s", ticker_u, e)

    pred_data = fetch_pred_data(ticker_u)

    if pred_data == "__MODEL_TRAINING__":
        return {
            "status": "training",
            "detail": f"model for {ticker} is being trained. retry after a few secs",
            "ticker": ticker_u,
        }

    if isinstance(pred_data, str):
        raise PipelineError(pred_data)

    try:
        forecast = (
            pred_data.get("predictions", {})
            .get("full_forecast", [])
        )
        history = pred_data.get("history", [])
        if not forecast:
            pred_str = f"no pred available for {ticker}"
        else:
            latest_close = float(history[-1]["close"]) if history else None
            next_week = pred_data.get("predictions", {}).get("next_week", {})
            week_low = float(next_week.get("low", 0.0)) if next_week else None
            week_high = float(next_week.get("high", 0.0)) if next_week else None

            lines = [f"5 day stock price forecast for {ticker}:"]
            if latest_close is not None:
                lines.append(f"Last close: Rs {latest_close:.2f}")
            if week_low is not None and week_high is not None:
                lines.append(f"Forecast range: Rs {week_low:.2f} to Rs {week_high:.2f}")
            for d in forecast[:5]:
                price = float(d.get("close", 0))
                lines.append(f"{d['date']}: Rs {price:.2f}")
            pred_str = "\n".join(lines)
    except Exception as e:
        raise PipelineError(f"Pred parsing failed {e}") from e

    graph = build_graph()
    state = {
        "preds": pred_str,
        "ticker": ticker_u,
        "messages": [HumanMessage(content=f"Start analysis {ticker_u}")],
    }
    config = {"configurable": {"thread_id": thread_id or "1"}}
    res = graph.invoke(state, config=config)

    final_report = res.get("final_report", "")
    summary = final_report.strip().split("\n\n", 1)[0].strip() if final_report else ""
    result = {
        "status": "completed",
        "ticker": ticker_u,
        "recommendation": res.get("recommendation", "NEUTRAL"),
        "confidence": res.get("confidence", "Medium"),
        "summary": summary,
        "final_report": final_report,
        "news_sentiment": res.get("news_sentiment", ""),
        "prediction": pred_data,
        "thread_id": thread_id,
        "use_fmi": use_fmi,
        "cached": False,
        "metadata": _build_metadata(
            ticker=ticker_u,
            source_cache="fresh",
            news_sentiment=res.get("news_sentiment", ""),
            embedder=embedder,
        ),
    }

    if embedder is not None and query_vec is not None and final_report:
        try:
            history = pred_data.get("history", [])
            last_price = float(history[-1]["close"]) if history else 0.0
            cache = SemanticCache(collection_name="analysis_cache")
            cache.save_episode(
                ticker=ticker_u,
                summary=summary,
                final_report=final_report,
                embedding=query_vec,
                recommendation=result["recommendation"],
                confidence=result["confidence"],
                last_price=last_price,
                prediction=pred_data,
                news_sentiment=result["news_sentiment"],
                thread_id=thread_id,
                use_fmi=use_fmi,
                llm_model=result["metadata"]["llm_model"],
                embed_model=result["metadata"].get("embed_model") or "",
                news_empty=result["metadata"]["news_empty"],
            )
            logger.info("semantic cache save succeeded for %s", ticker_u)
        except Exception as e:
            logger.exception("semantic cache save failed for %s: %s", ticker_u, e)

    return result
