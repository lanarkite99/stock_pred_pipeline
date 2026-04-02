import json
import os
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", "outputs"))

st.set_page_config(page_title="Stock Pred App", page_icon="ST", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top right, #1f2937 0%, #0f172a 45%, #020617 100%);
        color: #e5eef9;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0b1120 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }
    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 0.9rem;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    .stButton > button {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        color: #e5eef9;
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 10px;
    }
    .stButton > button:hover {
        border-color: #38bdf8;
        color: #f8fafc;
    }
    .panel-title {
        font-size: 0.95rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 0.6rem;
    }
    .status-chip {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.9rem;
    }
    .status-healthy { background: rgba(34, 197, 94, 0.18); color: #86efac; }
    .status-warning { background: rgba(245, 158, 11, 0.18); color: #fcd34d; }
    .status-critical { background: rgba(239, 68, 68, 0.18); color: #fca5a5; }
    .status-neutral { background: rgba(56, 189, 248, 0.18); color: #7dd3fc; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _post(path, payload=None):
    response = requests.post(f"{API_URL}{path}", json=payload or {}, timeout=120)
    response.raise_for_status()
    return response.json()



def _get(path: str):
    response = requests.get(f"{API_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()



def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None



def _task_status(ticker):
    try:
        return _get(f"/status/{ticker.lower()}")
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return None
        raise



def _ticker_dir(ticker):
    return OUTPUTS_DIR / ticker.lower()



def _analysis_path(ticker):
    return _ticker_dir(ticker) / "latest_analysis.json"



def _monitor_path(ticker):
    return _ticker_dir(ticker) / "monitor" / "latest_monitor.json"



def _status_path_text(data):
    if not data:
        return "No saved file yet"
    return data.get("status", "available")



def _status_class(status):
    status = (status or "").lower()
    if status in {"healthy", "completed"}:
        return "status-healthy"
    if status in {"warning", "running", "training"}:
        return "status-warning"
    if status in {"critical", "failed", "error", "offline"}:
        return "status-critical"
    return "status-neutral"



def _status_chip(label, status):
    st.markdown(
        f'<span class="status-chip {_status_class(status)}">{label}: {status or "unknown"}</span>',
        unsafe_allow_html=True,
    )



def _summary_table(items):
    def _display_value(value):
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return round(value, 4)
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        return str(value)

    return pd.DataFrame(
        [{"field": key.replace("_", " ").title(), "value": _display_value(value)} for key, value in items.items()]
    )



def _render_forecast_chart(full_forecast) -> None:
    frame = pd.DataFrame(full_forecast)
    if frame.empty or "date" not in frame.columns or "close" not in frame.columns:
        return

    frame["date"] = pd.to_datetime(frame["date"])
    min_close = float(frame["close"].min())
    max_close = float(frame["close"].max())
    span = max(max_close - min_close, max_close * 0.01, 1.0)
    padding = span * 0.25

    chart = (
        alt.Chart(frame)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=70, color="#38bdf8"), color="#38bdf8", strokeWidth=3)
        .encode(
            x=alt.X("date:T", title="date", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y(
                "close:Q",
                title="close",
                scale=alt.Scale(domain=[min_close - padding, max_close + padding], nice=False),
            ),
            tooltip=[alt.Tooltip("date:T", title="date"), alt.Tooltip("close:Q", title="close", format=",.2f")],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)



def _render_forecast_table(full_forecast):
    frame = pd.DataFrame(full_forecast)
    if frame.empty:
        return
    keep = [col for col in ["date", "open", "high", "low", "close", "volume"] if col in frame.columns]
    frame = frame[keep].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").round(2)
    st.dataframe(frame, use_container_width=True, hide_index=True)



def _render_check_table(title, values):
    if not values:
        return
    st.markdown(f"**{title}**")
    st.table(_summary_table(values))


st.title("Stock Pred Pipeline")
st.caption(f"FastAPI: {API_URL}")

with st.sidebar:
    st.subheader("Ticker")
    default_ticker = st.session_state.get("ticker", "INFY.NS")
    ticker = st.text_input("Yahoo ticker", value=default_ticker).strip().upper()
    st.session_state["ticker"] = ticker

    st.subheader("Actions")
    train_clicked = st.button("Train Child", use_container_width=True)
    predict_clicked = st.button("Predict", use_container_width=True)
    analyze_clicked = st.button("Analyze", use_container_width=True)
    monitor_clicked = st.button("Monitor", use_container_width=True)
    status_clicked = st.button("Check Status", use_container_width=True)

if not ticker:
    st.info("Enter a ticker to begin.")
    st.stop()

api_col, file_col = st.columns([1.0, 1.2])
latest_action = None

with api_col:
    st.markdown('<div class="panel-title">Live API</div>', unsafe_allow_html=True)

    try:
        health = _get("/health")
        backend_online = True
    except Exception as exc:
        backend_online = False
        health = {"status": "offline"}
        st.warning(f"Backend offline. Saved outputs are still available.\n\n{exc}")

    task_status = None
    if backend_online:
        try:
            task_status = _task_status(ticker)
        except Exception:
            task_status = {"status": "error"}

    metric_a, metric_b = st.columns(2)
    with metric_a:
        st.metric("API Health", health.get("status", "unknown"))
    with metric_b:
        st.metric("Task Status", task_status.get("status", "not found") if task_status else "not found")

    if train_clicked:
        try:
            latest_action = ("Train Child", _post("/train-child", {"ticker": ticker}))
            st.success("Training request sent.")
        except Exception as exc:
            st.error(f"Train failed: {exc}")

    if predict_clicked:
        try:
            latest_action = ("Predict", _post("/predict-child", {"ticker": ticker}))
            st.success("Prediction complete.")
        except Exception as exc:
            st.error(f"Predict failed: {exc}")

    if analyze_clicked:
        try:
            latest_action = ("Analyze", _post("/analyze", {"ticker": ticker}))
            st.success("Analysis complete.")
        except Exception as exc:
            st.error(f"Analyze failed: {exc}")

    if monitor_clicked:
        try:
            latest_action = ("Monitor", _post(f"/monitor/{ticker}"))
            st.success("Monitoring complete.")
        except Exception as exc:
            st.error(f"Monitor failed: {exc}")

    if status_clicked:
        try:
            result = _task_status(ticker)
            if result:
                latest_action = ("Task Status", result)
            else:
                st.info("No active training task found for this ticker.")
        except Exception as exc:
            st.error(f"Status lookup failed: {exc}")

    if latest_action:
        action_name, action_result = latest_action
        with st.expander(f"{action_name} response"):
            st.json(action_result)

with file_col:
    st.markdown('<div class="panel-title">Saved Outputs</div>', unsafe_allow_html=True)

    analysis_data = _read_json(_analysis_path(ticker))
    monitor_data = _read_json(_monitor_path(ticker))

    summary_a, summary_b, summary_c, summary_d = st.columns(4)
    with summary_a:
        st.metric("Latest Analysis", _status_path_text(analysis_data))
    with summary_b:
        st.metric("Latest Monitor", _status_path_text(monitor_data))
    with summary_c:
        if analysis_data:
            st.metric("Recommendation", analysis_data.get("recommendation", "n/a"))
        else:
            st.metric("Recommendation", "n/a")
    with summary_d:
        if analysis_data:
            st.metric("Confidence", analysis_data.get("confidence", "n/a"))
        else:
            st.metric("Confidence", "n/a")

    if analysis_data:
        st.markdown("**Analysis Summary**")
        _status_chip("Cache", analysis_data.get("metadata", {}).get("source_cache", "unknown"))
        st.write(analysis_data.get("summary", "No summary"))

        prediction = analysis_data.get("prediction", {})
        full_forecast = prediction.get("predictions", {}).get("full_forecast", [])
        if full_forecast:
            _render_forecast_chart(full_forecast)
            _render_forecast_table(full_forecast)

        analysis_meta = analysis_data.get("metadata", {})
        if analysis_meta:
            _render_check_table("Analysis Metadata", analysis_meta)

        with st.expander("Analysis JSON"):
            st.json(analysis_data)
    else:
        st.info("Run Analyze to create latest_analysis.json.")

    if monitor_data:
        st.markdown("**Monitor Summary**")
        summary = monitor_data.get("summary", {})
        _status_chip("Overall", monitor_data.get("status"))

        mon_a, mon_b, mon_c = st.columns(3)
        with mon_a:
            _status_chip("System", summary.get("system"))
        with mon_b:
            _status_chip("Regime", summary.get("regime"))
        with mon_c:
            _status_chip("Analysis", summary.get("analysis_quality"))

        _render_check_table("System Checks", monitor_data.get("system", {}))

        regime = monitor_data.get("regime", {})
        if regime.get("metrics"):
            _render_check_table("Regime Metrics", regime.get("metrics", {}))
        if regime.get("notes"):
            st.markdown("**Regime Notes**")
            for note in regime.get("notes", []):
                st.write(f"- {note}")

        quality = monitor_data.get("analysis_quality", {})
        if quality.get("checks"):
            _render_check_table("Analysis Checks", quality.get("checks", {}))

        with st.expander("Monitor JSON"):
            st.json(monitor_data)
    else:
        st.info("Run Monitor to create latest_monitor.json.")
