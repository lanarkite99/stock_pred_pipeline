# Stock Pred Pipeline

An end-to-end stock prediction and analysis system built with FastAPI, PyTorch, Redis, Chroma, Bedrock, Prometheus/Grafana, Feast, and Streamlit.

This repository combines:

- parent-child transfer learning for per-ticker forecasting
- agentic stock analysis with LangGraph-style orchestration and AWS Bedrock
- exact-cache and semantic-cache layers for faster repeat requests
- experiment tracking with MLflow and optional DagsHub integration
- custom monitoring for system health, market regime drift, and analysis quality
- local Docker Compose workflows and AWS EKS deployment assets

## Overview

`Stock Pred Pipeline` is designed as a compact MLOps-style workflow for stock prediction and analysis.

At a high level, the system can:

- train a reusable parent market model
- adapt child models for individual tickers
- serve short-horizon forecasts through FastAPI
- generate natural-language stock analysis reports using prediction outputs and recent news
- cache predictions in Redis and analysis results in Chroma
- monitor per-ticker health and save monitoring summaries
- expose runtime metrics to Prometheus and Grafana
- provide a Streamlit dashboard for running and inspecting the flow

## Key Capabilities

- **Transfer Learning**: parent-child training flow for ticker-specific forecasting
- **Prediction Serving**: FastAPI endpoints for training, prediction, analysis, and monitoring
- **Caching**: Redis exact-cache for prediction/task state and Chroma semantic cache for analysis reuse
- **Agentic Analysis**: Bedrock-backed report generation using prediction context and market news
- **Monitoring**: custom health, drift, and response-quality checks with saved monitor artifacts
- **Observability**: Prometheus metrics and Grafana dashboards
- **Tracking**: MLflow with optional DagsHub-backed remote tracking
- **Runtime Safeguards**: forecast sanity normalization, simple API rate limiting, and outlier statistics in training summaries

## Technical Design

```mermaid
flowchart TD
    U[User] --> S[Streamlit UI]
    U --> F[FastAPI Backend]
    S --> F

    F --> TR[Training Endpoints]
    F --> PR[Prediction Endpoints]
    F --> AN[Analysis Endpoint]
    F --> MO[Monitor Endpoint]
    F --> MT[/metrics]

    TR --> ML[Training Pipeline]
    PR --> IF[Inference Pipeline]
    AN --> AG[Agentic Analysis Layer]
    MO --> MON[Monitoring Runner]

    ML --> FEAST[Feast Local Repo + Online Features]
    ML --> ART[Artifacts]
    ML --> MLF[MLflow / DagsHub]

    PR --> R[(Redis)]
    AN --> R
    AN --> C[(Chroma)]

    AG --> BR[Amazon Bedrock]
    AG --> NEWS[News Sources]

    MT --> PM[Prometheus]
    PM --> GR[Grafana]

    ART --> OUT[outputs/<ticker>/...]
    MON --> OUT
```

## Technology Used

| Component | Technology |
|---|---|
| Backend API | FastAPI |
| Model Training / Inference | PyTorch (LSTM-based workflow) |
| Feature Layer | Feast config + local repo + optional Redis online store |
| Cache / Task State | Redis |
| Semantic Cache | Chroma |
| Agent Layer | LangGraph-style wrapper + Amazon Bedrock |
| News Sources | NewsAPI, Finnhub, Yahoo Finance fallback |
| Experiment Tracking | MLflow, optional DagsHub |
| Monitoring | Custom monitoring package |
| Observability | Prometheus, Grafana |
| Frontend | Streamlit |
| Local Orchestration | Docker Compose |
| Cloud Assets | Terraform, Kubernetes manifests, GitHub Actions |

## Repository Layout

```text
backend/         FastAPI app, schemas, state, task handling, rate limiting
feature_store/   Feast configuration and local registry/data
grafana/         Grafana provisioning and dashboards
k8s/             Kubernetes manifests for AWS/EKS deployment
logger/          Logging setup
monitoring/      System checks, regime drift, response quality, monitor runner
prometheus/      Prometheus scrape config
ref_docs/        Architecture and command reference docs
src/             Training, inference, agents, memory, utilities
streamlit_app/   Lightweight dashboard UI
terraform/       AWS infrastructure definitions
outputs/         Saved models, analysis JSON, monitor JSON, vector cache
```

## API Surface

Main endpoints:

- `GET /health`
- `GET /status/{task_id}`
- `GET /artifacts/{ticker}/analysis`
- `GET /artifacts/{ticker}/monitor`
- `POST /train-parent`
- `POST /train-child`
- `POST /predict-parent`
- `POST /predict-child`
- `POST /analyze`
- `POST /monitor/{ticker}`
- `GET /metrics`

Interactive docs:

- `http://localhost:8000/docs`

## Quick Start

### 1. Prerequisites

Make sure the following are available:

- Python 3.11+
- Docker and Docker Compose
- optional AWS credentials for Bedrock-backed analysis
- optional API keys for NewsAPI / Finnhub
- optional DagsHub + MLflow settings if remote experiment tracking is needed

### 2. Configure Environment

Create a `.env` file in the project root.

Typical variables:

```env
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
AWS_REGION=ap-south-1
BEDROCK_CHAT_MODEL_ID=openai.gpt-oss-20b-1:0
BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v1
NEWSAPI_KEY=
FMI_API_KEY=
USER_AGENT=stock-pred-pipeline/1.0
DAGSHUB_USER_NAME=
DAGSHUB_REPO_NAME=
DAGSHUB_TOKEN=
MLFLOW_TRACKING_URI=
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=admin
RATE_LIMIT_MAX_REQUESTS=60
RATE_LIMIT_WINDOW_SECONDS=60
```

### 3. Install dependencies locally

```powershell
uv sync
```

### 4. Start the local stack

```powershell
docker compose up -d --build fastapi redis prometheus grafana streamlit_app
```

Check service status:

```powershell
docker compose ps
```

## Service URLs

- FastAPI: `http://localhost:8000`
- FastAPI docs: `http://localhost:8000/docs`
- Streamlit: `http://localhost:8502`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Redis Stack UI: `http://localhost:8001`

Grafana default login:

- user: `admin`
- password: `admin` unless overridden in `.env`

## Example Flow

Train, predict, analyze, and monitor one ticker:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/train-child -ContentType "application/json" -Body '{"ticker":"INFY.NS"}'
Invoke-RestMethod http://127.0.0.1:8000/status/infy.ns | Format-List
Invoke-RestMethod -Method Post http://127.0.0.1:8000/predict-child -ContentType "application/json" -Body '{"ticker":"INFY.NS"}'
Invoke-RestMethod -Method Post http://127.0.0.1:8000/analyze -ContentType "application/json" -Body '{"ticker":"INFY.NS"}' | ConvertTo-Json -Depth 8
Invoke-RestMethod -Method Post http://127.0.0.1:8000/monitor/INFY.NS | ConvertTo-Json -Depth 8
```

Generated artifacts for a ticker include:

- `outputs/<ticker>/latest_analysis.json`
- `outputs/<ticker>/monitor/latest_monitor.json`
- `outputs/<ticker>/<TICKER>_child_model.pt`
- `outputs/<ticker>/<TICKER>_child_scaler.pkl`
- `outputs/<ticker>/<TICKER>_child_training_summary.json`

## Monitoring and Observability

The project has two distinct monitoring layers.

### Runtime observability

Prometheus and Grafana are used for:

- API request volume and latency
- cache-related metrics
- Redis and system health visibility
- dashboarding of service-level metrics

### Ticker-level monitoring

`POST /monitor/{ticker}` runs the custom monitoring package and writes a compact result with:

- `system` health
- `regime` drift / market-shift indicators
- `analysis_quality` checks

Saved artifact:

- `outputs/<ticker>/monitor/latest_monitor.json`

## Streamlit Dashboard

The Streamlit app is intentionally lightweight.

It can:

- trigger train / predict / analyze / monitor calls
- inspect saved analysis and monitoring artifacts
- render forecast charts
- present monitoring summaries in tables/cards

Run locally:

```powershell
uv run streamlit run streamlit_app/app.py
```

## AWS / EKS Notes

The repo also contains AWS deployment assets:

- Terraform under [terraform](./terraform)
- Kubernetes manifests under [k8s](./k8s)
- GitHub Actions workflows under [.github/workflows](./.github/workflows)

Current cloud-oriented behavior in code/manifests:

- FastAPI honors `ARTIFACT_DIR` so artifacts can live on mounted storage instead of only relative local paths.
- AWS manifests mount `/app/outputs` on an EBS-backed PVC.
- Prometheus and Grafana manifests support EBS-backed persistence through PVCs.
- Feast remains a local repo in the application, but the Kubernetes path now supports a Redis online store plus persistent `feature_store/data` storage on a PVC.

## Current Design Notes

- FastAPI currently runs with a single worker so in-process Prometheus metrics remain consistent.
- Redis is used for prediction exact cache, async task state, and simple rate limiting.
- Chroma is used for semantic caching of analysis outputs.
- Monitoring is custom domain logic, not Evidently-based reporting yet.
- Forecasts are normalized through sanity checks before being returned.
- Training summaries now include simple IQR-based outlier counts for visibility.

## Limitations

- `latest_analysis.json` stores the latest saved analysis, not a full timestamped evaluation history.
- `GET /status/{task_id}` is for async training status, not a generic artifact lookup endpoint.
- Streamlit is a thin client and depends on backend services for live actions.
- Regime drift is a warning signal, not an automatic retraining decision by itself.
- The current rate limiting is intentionally simple and IP-based, not a full auth-aware API gateway policy.
- Feast offline data is still repo-local logic, not a fully managed cloud-native feature platform.

## Documentation

Project-specific docs available in `ref_docs/`:

- [System Design](./ref_docs/system_design.md)
- [Cloud Design](./ref_docs/cloud.md)
- [Commands](./ref_docs/commands.md)

## License

Distributed under Apache License 2.0

## Contact

lets connect!
<a href="https://x.com/meetsiddhapura" target="_blank">
    <img src="https://cdn.simpleicons.org/x/white" alt="X logo" width="28" height="28" style="margin-right: 12px;">
</a>
<a href="https://linkedin.com/in/meet-siddhapura" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/>
</a>
<a href="https://lanarkite99.substack.com/" target="_blank">
    <img src="https://img.shields.io/badge/Substack-FF6719?style=for-the-badge&logo=substack&logoColor=white" alt="Substack"/>
</a>
