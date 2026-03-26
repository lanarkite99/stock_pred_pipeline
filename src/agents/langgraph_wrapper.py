from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph

from src.agents.agents import perf_analyst, report_generator
from src.agents.fetch import fetch_pred_data
from src.exception import PipelineError


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


def analyze_stock(ticker, thread_id=None, use_fmi=False):
    ticker_u = ticker.upper()
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

    return {
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
    }
