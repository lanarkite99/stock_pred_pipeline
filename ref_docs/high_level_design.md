# High Level Design

```mermaid
flowchart TD
    U[User] --> S[Streamlit App]
    U --> F[FastAPI API]

    S --> F

    subgraph Serving Layer
        F --> T[Training Endpoints]
        F --> P[Prediction Endpoints]
        F --> A[Analysis Endpoint]
        F --> M[Monitoring Endpoint]
    end

    subgraph ML Pipeline
        T --> FS[Feast Feature Store]
        T --> PT[PyTorch Parent and Child Training]
        PT --> ART[Model Artifacts and Outputs]

        P --> INF[Inference Pipeline]
        INF --> ART
    end

    subgraph Analysis Layer
        A --> AG[LangGraph Agent Flow]
        AG --> OL[Ollama LLM and Embeddings]
        AG --> NEWS[News Fetchers]
        NEWS --> FH[Finnhub]
        NEWS --> NA[NewsAPI]
        NEWS --> YF[Yahoo Finance]
    end

    subgraph Caching and State
        F --> R[Redis]
        A --> SC[Chroma Semantic Cache]
        P --> R
        M --> R
    end

    subgraph Monitoring and Observability
        F --> PM[/metrics]
        PM --> PR[Prometheus]
        PR --> G[Grafana]
        M --> MON[Custom Monitoring Runner]
        MON --> REG[Regime Drift Checks]
        MON --> QL[Analysis Quality Checks]
        MON --> OUT[latest_monitor.json]
    end

    subgraph Experiment Tracking
        T --> MF[MLflow]
        MF --> DH[DagsHub]
    end

    subgraph Artifacts
        AG --> LA[latest_analysis.json]
        ART --> OUTS[outputs/ticker]
        LA --> OUTS
        OUT --> OUTS
    end
```

## Notes

- `FastAPI` is the main entry point for training, prediction, analysis, monitoring, and Prometheus metrics.
- `Redis` stores exact-cache results and async task state.
- `Chroma` stores semantic cache entries for analysis reuse.
- `Ollama` powers both report generation and embedding-based semantic recall.
- `Prometheus` and `Grafana` cover system and API observability.
- `MLflow` logs runs locally or to `DagsHub` when configured.
- `Streamlit` is a thin UI on top of the FastAPI API plus saved output artifacts.
