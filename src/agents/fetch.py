import os
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

from src.exception import PipelineError

load_dotenv()

FINNHUB_API_KEY = os.getenv("FMI_API_KEY")
FINNHUB_URL = "https://finnhub.io/api/v1/company-news"
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

try:
    from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
except Exception:
    YahooFinanceNewsTool = None

try:
    import yfinance as yf
except Exception:
    yf = None


def fetch_pred_data(ticker):
    try:
        from src.pipelines.inference_pipeline import predict_child
        from backend.tasks import get_or_set_cache

        result, _ = get_or_set_cache(
            f"predict_child:{ticker.lower()}",
            lambda: predict_child(ticker),
            86400,
        )
        return result
    except PipelineError:
        raise
    except Exception as e:
        return f"failed to fetch prediction:{e}"


def get_stock_pred(ticker):
    data = fetch_pred_data(ticker)

    if isinstance(data, str):
        return data
    try:
        forecast = (
            data.get("predictions", {})
            .get("full_forecast", [])
        )
        if not forecast:
            return f"no prediction available for {ticker}"
        lines = [f"5 day stock price forecast for {ticker}:"]
        for d in forecast[:5]:
            price = float(d.get("close", 0))
            lines.append(f"{d['date']}: Rs {price:.2f}")

        return "\n".join(lines)
    except Exception as e:
        return f"Pred parsing failed {e}"


def _format_news_items(ticker: str, items: list[dict]) -> str:
    if not items:
        return ""

    valid_items = []
    for item in items[:4]:
        title = item.get("title") or item.get("headline") or ""
        summary = item.get("summary") or item.get("description") or ""
        link = item.get("link") or item.get("url") or ""
        if title.strip() or summary.strip() or link.strip():
            valid_items.append(item)

    if not valid_items:
        return f"No recent news with usable details was retrieved for {ticker}."

    res = [f"latest news for {ticker}:"]
    for item in valid_items:
        title = item.get("title") or item.get("headline") or "No headline"
        summary = item.get("summary") or item.get("description") or ""
        link = item.get("link") or item.get("url") or ""
        provider = item.get("publisher") or item.get("source") or ""
        published = item.get("providerPublishTime") or item.get("datetime")
        ts = ""
        if published:
            try:
                ts = datetime.utcfromtimestamp(int(published)).strftime("%Y-%m-%d")
            except Exception:
                ts = str(published)
        line = f"* {title}"
        if provider or ts:
            line += f" [{provider} {ts}]".rstrip()
        res.append(line)
        if summary:
            res.append(f"  {summary}")
        if link:
            res.append(f"  Link: {link}")
    return "\n".join(res)


def get_stock_news(ticker):
    try:
        end = datetime.utcnow().date()
        start = end - timedelta(days=7)

        data = requests.get(
            FINNHUB_URL,
            params={
                "symbol": ticker,
                "from": start.isoformat(),
                "to": end.isoformat(),
                "token": FINNHUB_API_KEY,
            },
            timeout=30,
        )
        if data.status_code == 200:
            news = data.json()[:4]
            if news:
                res = [f"latest news for {ticker} (Finnhub):"]
                for n in news:
                    ts = datetime.utcfromtimestamp(n.get("datetime", 0)).strftime("%Y-%m-%d")
                    res.append(
                        f"* {n.get('headline', 'No headline')} [{ts}]\n"
                        f"  {n.get('summary', '')}\n"
                        f"  Link: {n.get('url', '')}"
                    )
                return "\n".join(res)

        raise Exception("No output from Finnhub")
    except Exception:
        if yf is not None:
            try:
                news = yf.Ticker(ticker).news or []
                formatted = _format_news_items(ticker, news)
                if formatted:
                    return formatted
            except Exception:
                pass

        if not YahooFinanceNewsTool:
            return (
                f"No recent news was retrieved for {ticker}. "
                "Please verify current sector, macro, and geopolitical developments before investing."
            )
        try:
            yahoo_ticker = ticker.split(".")[0] if "." in ticker else ticker
            news = YahooFinanceNewsTool().invoke(yahoo_ticker)
            if isinstance(news, str) and "No news found" in news:
                return (
                    f"No recent news was retrieved for {ticker}. "
                    "Please verify current sector, macro, and geopolitical developments before investing."
                )
            return f"{ticker}:Latest news (yahoo):\n{news}"
        except Exception as e:
            return (
                f"No recent news was retrieved for {ticker} due to a news fetch issue ({e}). "
                "Please verify current sector, macro, and geopolitical developments before investing."
            )


fetch_list = [get_stock_pred, get_stock_news]



