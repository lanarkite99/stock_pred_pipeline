import os
import re

from langchain_core.messages import AIMessage, SystemMessage

from logger.logger import get_logger
from src.agents.fetch import get_stock_news, fetch_list

logger = get_logger()


llm = None
llm_init_error = None


def _get_llm():
    global llm, llm_init_error
    if llm is not None:
        return llm

    try:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gpt-oss:20b-cloud"),
            temperature=0.1,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ).bind_tools(fetch_list)
        return llm
    except Exception as e:
        llm_init_error = e
        logger.warning(f"llm init error {e}")
        raise


def _invoke_llm(prompt: str) -> str:
    response = _get_llm().invoke([SystemMessage(content=prompt)])
    return response.content if hasattr(response, "content") else str(response)


def _extract_recommendation(text: str) -> tuple[str, str]:
    upper = text.upper()
    recommendation = "NEUTRAL"
    confidence = "Medium"

    if "BULLISH" in upper:
        recommendation = "BULLISH"
    elif "BEARISH" in upper:
        recommendation = "BEARISH"

    if re.search(r"\bHIGH\b", upper):
        confidence = "High"
    elif re.search(r"\bLOW\b", upper):
        confidence = "Low"

    return recommendation, confidence


def perf_analyst(state):
    ticker = state["ticker"]
    preds = state.get("preds", "")
    logger.info(f"agent analyzing performance for {ticker}")

    if preds == "__MODEL_TRAINING__":
        logger.warning(f"[agent: performance] model for {ticker} is still training")
        return {
            "messages": [AIMessage(content=f"model for {ticker} is currently training.")],
            "preds": preds,
            "perf_summary": f"Model for {ticker} is currently training.",
        }

    prompt = f"""
You are a performance analyst for equities.
Analyze the 5-day stock forecast for {ticker}.

Forecast data:
{preds}

Return exactly 3 short bullets covering:
1. trend direction
2. expected price range or momentum
3. key caution or opportunity

Rules:
- Use only the numbers provided in the forecast data.
- Do not invent prices, targets, or percentages.
- Use INR / Rs notation, not dollars.
"""
    content = _invoke_llm(prompt)
    logger.info(f"DEBUG: Perf Output: {content[:100]}...")

    return {
        "messages": [AIMessage(content=content)],
        "preds": preds,
        "perf_summary": content,
    }


def market_expert(state):
    ticker = state["ticker"]
    news = get_stock_news(ticker)
    return {
        "messages": [AIMessage(content=news)],
        "news_sentiment": news,
    }


def report_generator(state):
    ticker = state["ticker"]
    preds = state.get("preds", "")
    perf_summary = state.get("perf_summary", "")
    logger.info(f"agent generating report for {ticker}")
    news = get_stock_news(ticker)

    prompt = f"""
You are a senior market analyst and editor.
Write a concise professional markdown report for {ticker}.

Forecast data:
{preds}

Performance analyst summary:
{perf_summary}

Latest news and sentiment:
{news}

Requirements:
- Use INR/rupees, not dollars.
- Keep it concise and professional.
- Do not use tables.
- Keep the report readable in plain text and JSON viewers.
- Keep Executive Summary to 2 short sentences.
- Keep Forecast Outlook to 3 bullets max.
- Keep News Sentiment to 3 bullets max.
- Mention geopolitical context only if it is explicitly supported by the news input.
- If recent news is unavailable or weak, explicitly say the news signal is limited and investors should verify current sector, macro, and geopolitical developments before investing.
- Include sections: Executive Summary, Forecast Outlook, News Sentiment, Recommendation.
- End with exactly:
  Market Stance: BULLISH/BEARISH/NEUTRAL
  Confidence: High/Medium/Low
- Ensure the recommendation is logically consistent with the forecast and news.
- Use only the prices and ranges provided in the forecast input. Do not invent new price levels.
"""
    final_text = _invoke_llm(prompt)
    recommendation, confidence = _extract_recommendation(final_text)

    return {
        "messages": [AIMessage(content=final_text)],
        "news_sentiment": news,
        "final_report": final_text,
        "recommendation": recommendation,
        "confidence": confidence,
    }


def critic_agent(state):
    return {
        "messages": [],
        "final_report": state.get("final_report", ""),
    }
